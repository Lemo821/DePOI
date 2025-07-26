import copy
import os.path
import pickle
import sys
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Testdataset


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def calculate_laplacian_matrix(adj_mat):
    n_vertex = adj_mat.shape[0]
    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row
    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))
    wid_deg_mat = deg_mat + id_mat
    wid_adj_mat = adj_mat + id_mat
    hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
    return hat_rw_normd_lap_mat


def compute_time(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def time_interval_mat(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum + 1)):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1:
                break
        data_train[user] = compute_time(time_seq, time_span)
    return data_train


def compute_dist(dis_seq, dis_span):
    dis_span = dis_span
    size = len(dis_seq)
    dis_matrix = np.zeros([size, size], dtype=np.float64)
    for i in range(size):
        for j in range(size):
            lon1 = float(dis_seq[i].split(',')[0])
            lat1 = float(dis_seq[i].split(',')[1])
            lon2 = float(dis_seq[j].split(',')[0])
            lat2 = float(dis_seq[j].split(',')[1])
            span = int(abs(haversine(lon1, lat1, lon2, lat2)))
            if dis_seq[i] == '0,0' or dis_seq[j] == '0,0':
                dis_matrix[i][j] = dis_span
            elif span > dis_span:
                dis_matrix[i][j] = dis_span
            else:
                dis_matrix[i][j] = span
    return dis_matrix


def dist_interval_mat(user_train, usernum, maxlen, dis_span):
    data_train = dict()
    for user in tqdm(range(1, usernum + 1)):
        dis_seq = ['0,0'] * maxlen
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            dis_seq[idx] = i[2]
            idx -= 1
            if idx == -1:
                break
        data_train[user] = compute_dist(dis_seq, dis_span)
    return data_train


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def clean_sort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]], x[2]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1), x[2]], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)


def data_partition(fname):
    User = defaultdict(list)
    user_train, user_valid, user_test = {}, {}, {}

    print('Starting data partition on {}.'.format(fname.split('/')[1]))
    f = open(fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, location, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u, i = int(u), int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open(fname, 'r')

    for line in f:
        try:
            u, i, location, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        time_set.add(timestamp)
        User[u].append([i, timestamp, location])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = clean_sort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    pop_freq = [0] * (itemnum+1)
    for u in user_train:
        history = user_train[u]
        for elem in history:
            pop_freq[elem[0] - 1] += 1
    return [user_train, user_valid, user_test, usernum, itemnum, timenum, pop_freq]


def sparse_dropout(x, rate, noise_shape):
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).bool()
    i = x._indices()
    v = x._values()

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    out = out * (1. / (1 - rate))

    return out


def get_adj_matrix(matrix):
    row_sum = np.array(matrix.sum(1)) + 1e-24
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(matrix.dot(degree_mat_inv_sqrt)).todense()
    return rel_matrix_normalized


def generate_test(dataset, args):
    [train, valid, test, usernum, _, _,_] = copy.deepcopy(dataset)
    users = range(1, usernum + 1)
    all_test_user, all_test_seq, all_test_time_matrix, all_test_dis_matrix, all_labels = [], [], [], [], []

    for u in users:
        if u % 1000 == 0:
            print('.', end='')
            sys.stdout.flush()
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        dis_seq = ['0,0'] * args.maxlen
        idx = args.maxlen - 1
        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        dis_seq[idx] = valid[u][0][2]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx], time_seq[idx], dis_seq[idx] = i[0], i[1], i[2]
            idx -= 1
            if idx == -1:
                break
        time_matrix = compute_time(time_seq, args.time_span)
        dis_matrix = compute_dist(dis_seq, args.dis_span)
        all_test_user.append(u)
        all_test_seq.append(seq)
        all_test_time_matrix.append(time_matrix)
        all_test_dis_matrix.append(dis_matrix)
        all_labels.append(test[u][0][0])

    with open(args.dataset + '_test_instance.pkl', 'wb') as f:
        pickle.dump(all_test_user, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_test_seq, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_test_time_matrix, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_test_dis_matrix, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_labels, f, pickle.HIGHEST_PROTOCOL)


def evaluate_test(model, dataset, args):
    HT, NDCG = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    test_user_num = 0.0
    test_pkl_path = args.dataset + "_test_instance.pkl"
    if not os.path.exists(test_pkl_path):
        print('Preparing test instances...')
        generate_test(dataset, args)
    with open(test_pkl_path, 'rb') as f:
        all_u = pickle.load(f)
        all_seqs = pickle.load(f)
        all_time_matrix = pickle.load(f)
        all_distance_matrix = pickle.load(f)
        all_labels = pickle.load(f)

    test_dataset = Testdataset(all_seqs, all_time_matrix, all_distance_matrix, all_labels)
    dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    for instance in dataloader:
        sys.stdout.flush()
        u, seq, time_matrix, dis_matrix, label = instance
        predictions, no_use = model.predict(u, seq, time_matrix, dis_matrix, [1])
        predictions = -predictions
        ranks = predictions.argsort().argsort().cpu()

        rank = []
        for i in range(len(ranks)):
            rank.append(ranks[i, label[i]])
        test_user_num += len(rank)
        for i in rank:
            if i < 2:
                NDCG[0] += 1 / np.log2(i + 2)
                HT[0] += 1
            if i < 5:
                NDCG[1] += 1 / np.log2(i + 2)
                HT[1] += 1
            if i < 10:
                NDCG[2] += 1 / np.log2(i + 2)
                HT[2] += 1

    return [x / test_user_num for x in NDCG], [x / test_user_num for x in HT]
