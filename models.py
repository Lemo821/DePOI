import sys

import torch
import torch.nn as nn

from DeGCN import DeGCN

FLOAT_MIN = -sys.float_info.max


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = nn.Linear(hidden_size, hidden_size)
        self.K_w = nn.Linear(hidden_size, hidden_size)
        self.V_w = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, dis_matrix_K, dis_matrix_V,
                abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        dis_matrix_K_ = torch.cat(torch.split(dis_matrix_K, self.head_size, dim=3), dim=0)
        dis_matrix_V_ = torch.cat(torch.split(dis_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)
        attn_weights += dis_matrix_K_.matmul((Q_.unsqueeze(-1))).squeeze(-1)

        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) * (-2 ** 32 + 1)
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)
        outputs += attn_weights.unsqueeze(2).matmul(dis_matrix_V_).reshape(outputs.shape).squeeze(2)

        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)

        return outputs


class Contrastive_BPR(nn.Module):
    def __init__(self, beta=1):
        super(Contrastive_BPR, self).__init__()
        self.Activation = nn.Softplus(beta=beta)

    def forward(self, x, pos, neg):
        loss_logit = (x * neg).sum(-1) - (x * pos).sum(-1)
        return self.Activation(loss_logit)


# main model
class DePOI(torch.nn.Module):
    def __init__(self, user_num, item_num, tran_mat, dist_mat, args):
        super(DePOI, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.tran_mat = tran_mat
        self.dist_mat = dist_mat
        self.device = args.device
        self.geo_weight = args.geo_weight

        self.tran_item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.geo_item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.item_emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.causal_item_embs = None
        self.bias_item_embs = None

        self.tran_proj = nn.Linear(args.hidden_units, args.hidden_units, bias=False)
        self.geo_proj = nn.Linear(args.hidden_units, args.hidden_units, bias=False)
        self.causal_proj = nn.Linear(args.hidden_units, args.hidden_units, bias=False)
        self.bias_proj = nn.Linear(args.hidden_units, args.hidden_units, bias=False)

        self.CL_builder = Contrastive_BPR()

        self.transition_gcn = DeGCN(input_dim=args.hidden_units,
                                    output_dim=args.hidden_units,
                                    node_num=self.item_num,
                                    anchor_num=args.anchor_num,
                                    layer=args.tran_gcn_layer,
                                    dropout=args.dropout_rate,
                                    device=args.device)

        self.geography_gcn = DeGCN(input_dim=args.hidden_units,
                                   output_dim=args.hidden_units,
                                   node_num=self.item_num,
                                   anchor_num=args.anchor_num,
                                   layer=args.geo_gcn_layer,
                                   dropout=args.dropout_rate,
                                   device=args.device)

        self.abs_pos_K_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.time_matrix_K_emb = nn.Embedding(args.time_span + 1, args.hidden_units)
        self.time_matrix_V_emb = nn.Embedding(args.time_span + 1, args.hidden_units)

        self.dis_matrix_K_emb = nn.Embedding(args.dis_span + 1, args.hidden_units)
        self.dis_matrix_V_emb = nn.Embedding(args.dis_span + 1, args.hidden_units)

        self.abs_pos_K_emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = nn.Dropout(p=args.dropout_rate)

        self.dis_matrix_K_dropout = nn.Dropout(p=args.dropout_rate)
        self.dis_matrix_V_dropout = nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units, args.num_heads,
                                                         args.dropout_rate, args.device)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def seq2feats(self, user_ids, log_seqs, time_matrices, dis_matrices, item_embs):
        seqs = item_embs[torch.LongTensor(log_seqs).to(self.device), :]
        seqs *= item_embs.shape[1] ** 0.5
        seqs = self.item_emb_dropout(seqs)

        positions = torch.arange(log_seqs.shape[1]).unsqueeze(0).expand(log_seqs.shape[0], -1)
        positions = torch.LongTensor(positions).to(self.device)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = torch.LongTensor(time_matrices).to(self.device)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        dis_matrices = torch.LongTensor(dis_matrices).to(self.device)
        dis_matrix_K = self.dis_matrix_K_emb(dis_matrices)
        dis_matrix_V = self.dis_matrix_V_emb(dis_matrices)
        dis_matrix_K = self.dis_matrix_K_dropout(dis_matrix_K)
        dis_matrix_V = self.dis_matrix_V_dropout(dis_matrix_V)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs,
                                                   timeline_mask, attention_mask,
                                                   time_matrix_K, time_matrix_V,
                                                   dis_matrix_K, dis_matrix_V,
                                                   abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices, dis_matrices, pos_seqs, neg_seqs, anchor_idx):
        anchor_idx = anchor_idx.to(self.device)
        self.anchor_idx = anchor_idx

        causal_tran_embs, bias_tran_embs, tran_support_loss = self.transition_gcn(self.tran_item_emb, self.anchor_idx,
                                                                                  self.tran_mat)
        causal_geo_embs, bias_geo_embs, geo_support_loss = self.geography_gcn(self.geo_item_emb, self.anchor_idx,
                                                                              self.dist_mat)

        tran_pool = torch.mean(causal_tran_embs, dim=0, keepdim=True).repeat(causal_tran_embs.shape[0], 1)
        geo_pool = torch.mean(causal_geo_embs, dim=0, keepdim=True).repeat(causal_geo_embs.shape[0], 1)
        tran_pool = self.tran_proj(tran_pool.to(self.device))
        geo_pool = self.geo_proj(geo_pool.to(self.device))
        con_loss = (self.CL_builder(causal_tran_embs, tran_pool, geo_pool) +
                    self.CL_builder(causal_geo_embs, geo_pool, tran_pool))

        causal_item_embs = causal_tran_embs + causal_geo_embs * self.geo_weight
        bias_item_embs = bias_tran_embs + bias_geo_embs * self.geo_weight
        self.causal_item_embs = causal_item_embs
        self.bias_item_embs = bias_item_embs

        causal_pool = torch.mean(causal_item_embs, dim=0, keepdim=True).repeat(causal_item_embs.shape[0], 1)
        bias_pool = torch.mean(bias_item_embs, dim=0, keepdim=True).repeat(bias_item_embs.shape[0], 1)
        norm_c_square = torch.sum((causal_item_embs - causal_pool) ** 2, dim=1)
        norm_b_square = torch.sum((causal_item_embs - bias_pool) ** 2, dim=1)
        loss_per_item = (norm_c_square - norm_b_square) ** 2 + 1e-6
        cb_con_loss = torch.sum(loss_per_item)

        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, causal_item_embs)
        log_feats_bias = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, bias_item_embs)

        pos_embs = causal_item_embs[torch.LongTensor(pos_seqs).to(self.device), :]
        neg_embs = causal_item_embs[torch.LongTensor(neg_seqs).to(self.device), :]
        pos_embs_bias = bias_item_embs[torch.LongTensor(pos_seqs).to(self.device), :]

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        pos_logits_bias = (log_feats_bias * pos_embs_bias).sum(dim=-1)

        fin_logits = log_feats.matmul(causal_item_embs.transpose(0, 1))
        fin_logits = fin_logits.reshape(-1, fin_logits.shape[-1])
        fin_logits_bias = log_feats_bias.matmul(bias_item_embs.transpose(0, 1))
        fin_logits_bias = fin_logits_bias.reshape(-1, fin_logits_bias.shape[-1])

        return (pos_logits, neg_logits, fin_logits, self.causal_item_embs[0],
                tran_support_loss, geo_support_loss, causal_item_embs, bias_item_embs,
                pos_logits_bias, fin_logits_bias, con_loss, cb_con_loss)

    def predict(self, user_ids, log_seqs, time_matrices, dis_matrices, item_indices):
        poi_transition_emb, _, _ = self.transition_gcn(self.tran_item_emb, self.anchor_idx, self.tran_mat)
        poi_geography_emb, _, _ = self.geography_gcn(self.geo_item_emb, self.anchor_idx, self.dist_mat)
        # only use causal rep for prediction
        item_embs = poi_transition_emb + poi_geography_emb * self.geo_weight
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, item_embs)

        final_feat = log_feats[:, -1, :]

        item_emb = item_embs

        logits = final_feat.matmul(item_emb.transpose(0, 1))

        return logits, item_indices
