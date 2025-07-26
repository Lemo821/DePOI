import argparse
import gc
import os
import time
import random
from datetime import datetime

from dataset import Traindataset
from models import DePOI
from utils import *
import torch.nn.functional as F


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='nyc')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=64, type=int, help='Embedding size.')
parser.add_argument('--num_blocks', default=1, type=int, help='Number of stacked attention layer.')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--output_epochs', default=5, type=int)
parser.add_argument('--num_heads', default=1, type=int, help='Self attention heads')
parser.add_argument('--tran_gcn_layer', default=1, type=int)
parser.add_argument('--geo_gcn_layer', default=1, type=int)
parser.add_argument('--geo_weight', default=1, type=float)
parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0001, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--device_id', default='1', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--laplacian', default=True, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--time_span', default=256, type=int)
parser.add_argument('--dis_span', default=256, type=int)
parser.add_argument('--tran_reg', default=1.0, type=float)
parser.add_argument('--geo_reg', default=1.0, type=float)
parser.add_argument('--kl_reg', default=1.0, type=float)
parser.add_argument('--cb_reg_loss', default=0.01, type=float, help='Strength of causal-bias disagreement reg')
parser.add_argument('--tg_reg_loss', default=0.01, type=float, help='Strength of trans-geo contrastive loss')
parser.add_argument('--anchor_num', default=1000, type=int)

args = parser.parse_args()


def mask(adj, epsilon=0, mask_value=-1e16):
    mask = (adj > epsilon).detach().float()
    update_adj = adj * mask + (1 - mask) * mask_value
    return update_adj


if __name__ == '__main__':
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))

    setup_seed(args.seed)

    time_string = datetime.now().strftime("%m%d%H%M%S")
    log_path = os.path.join('log', args.dataset + '_' + time_string + '.log')

    # data partition
    dataset = data_partition('data/' + args.dataset + '/' + args.dataset + '.txt')
    [user_train, user_valid, user_test, usernum, itemnum, timenum, pop_freq] = dataset
    num_batch = len(user_train) // args.batch_size

    # load transition distribution matrix
    npz_path = 'data/' + args.dataset + '/' + args.dataset + '_mat.npz'
    tra_adj_matrix = sp.load_npz(npz_path)
    tra_adj_matrix = tra_adj_matrix.todok()

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('Batch number: {}. Average sequence length: {:.2f}'.format(num_batch, cc / len(user_train)))
    print('After filtering, #POI: {}, #user: {}'.format(itemnum, usernum))

    with open(os.path.join('log', args.dataset + '_' + time_string + '.log'), 'a+') as f:
        for k, v in sorted(vars(args).items()):
            f.write('{}: {}\n'.format(k, v))

    # generate or load spatio-temporal interval matrix
    mat_filename = 'data/{}/relation_matrix_{}_{}_{}.pickle'.format(
        args.dataset, args.dataset, args.maxlen, args.time_span)
    if os.path.exists(mat_filename):
        relation_matrix = pickle.load(open(mat_filename, 'rb'))
    else:
        relation_matrix = time_interval_mat(user_train, usernum, args.maxlen, args.time_span)
        pickle.dump(relation_matrix, open(mat_filename, 'wb'))

    dis_mat_filename = 'data/{}/relation_dis_matrix_{}_{}_{}.pickle'.format(
        args.dataset, args.dataset, args.maxlen, args.dis_span)
    if os.path.exists(dis_mat_filename):
        dis_relation_matrix = pickle.load(open(dis_mat_filename, 'rb'))
    else:
        dis_relation_matrix = dist_interval_mat(user_train, usernum, args.maxlen, args.dis_span)
        pickle.dump(dis_relation_matrix, open(dis_mat_filename, 'wb'))

    print('Done with spatio-temporal interval matrix.')

    tran_mat_npz = np.load('data/{}/{}_tran_mat.npz'.format(args.dataset, args.dataset))['data']
    dist_mat_npz = np.load('data/{}/{}_dist_mat.npz'.format(args.dataset, args.dataset))['data']
    dist_mat_npz = dist_mat_npz - np.eye(dist_mat_npz.shape[0])

    if args.laplacian:
        tran_mat = torch.FloatTensor(calculate_laplacian_matrix(tran_mat_npz)).to(args.device)
        dist_mat = torch.FloatTensor(calculate_laplacian_matrix(dist_mat_npz)).to(args.device)
    else:
        tran_mat = torch.FloatTensor(tran_mat_npz).to(args.device)
        dist_mat = torch.FloatTensor(dist_mat_npz).to(args.device)

    print('Done with transition and geography graph matrix.')

    # load data
    train_dataset = Traindataset(user_train, relation_matrix, dis_relation_matrix, itemnum, args.maxlen)
    dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)
    # create model
    model = DePOI(usernum, itemnum, tran_mat, dist_mat, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass  # just ignore those failed init layers

    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)

    ce_criterion = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]
    adam_optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)

    T = 0.0
    t0 = time.time()
    anchor_num = args.anchor_num

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        print('Training on Epoch-{}'.format(epoch), end=' ')
        anchor_idx = torch.randperm(itemnum)[:anchor_num]
        tra_adj_matrix_anchor = tra_adj_matrix[anchor_idx.numpy(), :].todense()
        prior = torch.FloatTensor(tra_adj_matrix_anchor).to(args.device)
        anchor_idx += 1
        if args.inference_only:
            break
        for step, instance in enumerate(dataloader):
            u, seq, time_seq, pos, neg, time_matrix, dis_matrix = instance
            (pos_logits, neg_logits, fin_logits, padding, tran_support, geo_support, causal_emb,
             bias_emb, pos_logits_bias, fin_logits_bias, tg_con_loss, cb_con_loss) = (
                model(u, seq, time_matrix, dis_matrix, pos, neg, anchor_idx))

            pos_labels, neg_labels = (torch.ones(pos_logits.shape, device=args.device),
                                      torch.zeros(neg_logits.shape, device=args.device))
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            pos_label_for_crosse = pos.numpy().reshape(-1)

            indices_for_crosse = np.where(pos_label_for_crosse != 0)
            pos_label_cross = torch.tensor(pos_label_for_crosse[indices_for_crosse], device=args.device)

            main_loss = ce_criterion(fin_logits[indices_for_crosse], pos_label_cross.long())

            probabilities = torch.softmax(fin_logits_bias[indices_for_crosse], dim=1)
            pop_freq_tensor = torch.FloatTensor(pop_freq).unsqueeze(0)
            scaled_pop_freq_tensor = pop_freq_tensor
            pop_distribution = scaled_pop_freq_tensor.repeat(probabilities.shape[0], 1).to(args.device)
            bias_loss = kl_loss(torch.log(mask(probabilities) + 1e-9),
                                torch.softmax(mask(pop_distribution) + 1e-9, dim=-1)).mean()

            loss = (main_loss +
                    cb_con_loss * args.cb_reg_loss +
                    tg_con_loss * args.tg_reg_loss +
                    bias_loss * args.kl_reg)

            loss.backward()
            adam_optimizer.step()

        print('loss: {:.4f}, bias loss: {:.4f}'.format(loss, bias_loss))
        print('loss: {:.4f}'.format(loss))

        if epoch % args.output_epochs == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1

            before_test = time.time()
            NDCG, HR = evaluate_test(model, dataset, args)
            print(f'Evaluating on Epoch-{epoch}, time: {T:.2f}s, loss: {loss:.4f}')
            print(f'HitR@2 = {HR[0]:.4f}, HitR@5 = {HR[1]:.4f}, HitR@10 = {HR[2]:.4f}')
            print(f'NDCG@2 = {NDCG[0]:.4f}, NDCG@5 = {NDCG[1]:.4f}, NDCG@10 = {NDCG[2]:.4f}')
            infer_time = time.time() - before_test
            print(f'Inference time: {infer_time:.2f}s')
            print('-' * 60)
            with open(os.path.join('log', args.dataset + '_' + time_string + '.log'), 'a+') as f:
                f.write(f'Epoch:{epoch + 1}, loss: {loss:.4f}\nHitR: {HR}, NDCG: {NDCG}\n')
                f.write('-' * 60 + '\n')
            t0 = time.time()
            model.train()
        gc.collect()

    print("=====FIN=====")
