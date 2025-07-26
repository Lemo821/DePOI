import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

device = 'cuda'


class DeGCN(nn.Module):
    def __init__(self, input_dim, output_dim, node_num, anchor_num,layer=1, dropout=0.2, device='cuda', is_sparse_inputs=False,
                 bias=False):
        super(DeGCN, self).__init__()
        self.embedding_size = input_dim
        self.dropout = dropout
        self.layer_num = layer
        self.is_sparse_inputs = is_sparse_inputs
        self.device = device
        self.similarity_weight = Parameter(torch.nn.init.xavier_uniform_(torch.empty(input_dim, output_dim)))
        self.attention_matrix = Parameter(torch.nn.init.xavier_uniform_(torch.empty(input_dim, anchor_num)))
        self.a = Parameter(torch.nn.init.xavier_uniform_(torch.empty(node_num, 2 * node_num)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        self.bn = nn.BatchNorm1d(output_dim)

    def cosine_matrix_div(self, emb, anchor):
        node_norm = emb.div(torch.norm(emb, p=2, dim=-1, keepdim=True))
        anchor_norm = anchor.div(torch.norm(anchor, p=2, dim=-1, keepdim=True))
        cos_adj = torch.mm(node_norm, anchor_norm.transpose(-1, -2))
        return cos_adj

    def get_neighbor_hard_threshold(self, adj, epsilon=0, mask_value=0):
        mask = (adj > epsilon).detach().float()
        update_adj = adj * mask + (1 - mask) * mask_value
        return update_adj


    def masking_matrix(self, emb, adj):
        Whi = torch.matmul(emb, self.attention_matrix)
        Whj = torch.matmul(emb, self.attention_matrix)
        e = torch.matmul(self.a, torch.cat([Whi, Whj], dim=0))
        masking_matrix = torch.sigmoid(e)
        torch.cuda.empty_cache()
        causal_adj = masking_matrix * adj
        bias_adj = (1 - masking_matrix) * adj

        return causal_adj, bias_adj

    def forward(self, inputs, anchor_idx, adj):
        x = inputs.weight[1:, :]
        anchor = inputs(anchor_idx)
        anchor_adj = self.cosine_matrix_div(x, anchor)
        anchor_adj = self.get_neighbor_hard_threshold(anchor_adj)

        anchor_causal_adj, anchor_bias_adj = self.masking_matrix(x, anchor_adj)

        causal_x_fin, bias_x_fin = [x], [x]
        causal_layer, bias_layer = x, x

        for f in range(self.layer_num):
            node_norm = anchor_causal_adj / torch.clamp(torch.sum(anchor_causal_adj, dim=-2, keepdim=True), min=1e-12)
            anchor_norm = anchor_causal_adj / torch.clamp(torch.sum(anchor_causal_adj, dim=-1, keepdim=True), min=1e-12)
            causal_layer = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), causal_layer)) + causal_layer
            causal_x_fin += [causal_layer]
        causal_x_fin = torch.stack(causal_x_fin, dim=1)
        causal_out = torch.sum(causal_x_fin, dim=1)
        mp2 = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), causal_out], dim=0)

        for f in range(self.layer_num):
            node_norm = anchor_bias_adj / torch.clamp(torch.sum(anchor_bias_adj, dim=-2, keepdim=True), min=1e-12)
            anchor_norm = anchor_bias_adj / torch.clamp(torch.sum(anchor_bias_adj, dim=-1, keepdim=True), min=1e-12)
            bias_layer = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), bias_layer)) + bias_layer
            bias_x_fin += [bias_layer]

        bias_x_fin = torch.stack(bias_x_fin, dim=1)
        bias_out = torch.sum(bias_x_fin, dim=1)
        mp2_bias = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), bias_out], dim=0)

        return mp2, mp2_bias, anchor_adj
