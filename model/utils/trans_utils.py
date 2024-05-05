import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
import scipy.sparse.csgraph as csgraph
  
class TemporalNodeFeature(nn.Module):
    """
    We assume that input shape is B, T
        - Only contains temporal information with index
    Arguments:
        - vocab_size: total number of temporal features (e.g., 7 days)
        - freq_act: periodic activation function
        - n_freq: number of hidden elements for frequency components
            - if 0 or H, it only uses linear or frequency component, respectively
    """
    def __init__(self, hidden_size, vocab_size, scaler=1, steps_per_day=24, freq_act = torch.sin, n_freq = 1):
        super(TemporalNodeFeature, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.freq_act = freq_act
        self.n_freq = n_freq
        self.scaler = scaler
        self.steps_per_day = steps_per_day

    def forward(self, tod, dow):
        # args: x [B, T]
        # return: [B, T, D]
        x = (tod*self.scaler).int() + dow*self.steps_per_day
        x_emb = self.embedding(x.long())
        x_weight = self.linear(x_emb)
        if self.n_freq == 0:
            return x_weight
        if self.n_freq == x_emb.size(-1):
            return self.freq_act(x_weight)
        x_linear = x_weight[...,self.n_freq:]
        x_act = self.freq_act(x_weight[...,:self.n_freq])
        return torch.cat([x_linear, x_act], dim = -1)

class SpatialNodeFeature(nn.Module):
    def __init__(self, num_degree, d_model):
        super(SpatialNodeFeature, self).__init__()
        self.degree_encoder = nn.Embedding(num_degree, d_model)

    def forward(self, degree):
        # args: degree [n]
        # return: [n, D]
        degree_feature = self.degree_encoder(degree) # [n, d]
        return degree_feature

'''
1. floyd_warshall算法的要求
    + 矩阵的对角线元素应该设置为零，表示从任何节点到其自身的距离为零
    + 矩阵中的元素代表从一个节点到另一个节点的边的权重。
    + 如果两个节点没有直接连接，相应的矩阵元素通常为numpy.inf
'''

class SpatialAttnBias(nn.Module):
    def __init__(self, num_shortpath, bias_dim=1):
        super(SpatialAttnBias, self).__init__()
        self.bias_dim = bias_dim
        self.num_shortpath = num_shortpath
        self.attn_bias_encoder = nn.Embedding(num_shortpath, bias_dim)
        
    def forward(self, graph, dataset):
        num_shortpath, shortest_path = get_shortpath_num(graph, dataset)
        assert self.num_shortpath == num_shortpath, "num_shortpath结果不相等"
        attn_bias_spatial = self.attn_bias_encoder(shortest_path)
        return attn_bias_spatial

def get_shortpath_num(graph, dataset):
    if graph.is_cuda:
        graph = graph.cpu().numpy()
    else:
        graph = graph.numpy()
    if dataset in ['METRLA', 'PEMSBAY']:
        graph = modify_graph_metr(graph)
    else:
        graph = modify_graph_nyc(graph)
        
    shortest_path = csgraph.floyd_warshall(graph, return_predecessors=False)
    
    if dataset in ['METRLA', 'PEMSBAY']:
        shortest_path = np.where(np.isinf(shortest_path), -1, shortest_path * 100)
        shortest_path = shortest_path + 1

    shortest_path = torch.tensor(shortest_path, dtype=torch.int, device='cuda')
    num_shortpath = int(torch.max(shortest_path)) + 1
    return num_shortpath, shortest_path

# for METRLA/PEMSBAY dataset
# metr graph原形式：见DCRNN
def modify_graph_metr(graph):
    adj_matrix = np.array(graph, dtype=float)
    non_zero_mask = adj_matrix != 0
    adj_matrix[non_zero_mask] = np.sqrt(np.abs(np.log(adj_matrix[non_zero_mask])))
    # 将原始矩阵中为0的元素设置为无穷大
    adj_matrix[~non_zero_mask] = np.inf
    # 将对角线上的元素设置为0
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix
 
# Replace 0s with float('inf'), except on the diagonal for NYC* dataset
# NYC* graph原形式： 对角线为0，其他地方1-连接，0-不连接
def modify_graph_nyc(graph):
    adj_matrix = np.array(graph, dtype=float)
    adj_matrix[adj_matrix == 0] = np.inf
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix

def get_num_degree(graph):
    degree = get_degree_array(graph)
    num_degree = int(torch.max(degree)) + 1
    return num_degree

def get_degree_array(graph):
    dge_graph = (graph != 0).int()
    torch.diagonal(dge_graph)[:] = 0
    degree = dge_graph.sum(dim=-1)
    return degree

def generate_moe_posList(layers, moe_position):
    length = len(layers)
    if moe_position == 'Full':
        return [1] * length
    elif moe_position == 'Half':
        half_length = length // 2
        return [0] * half_length + [1] * (length - half_length)
    elif moe_position == 'woS':
        return [0 if layer == 'S' else 1 for layer in layers]
    elif moe_position == 'woT':
        return [0 if layer == 'T' else 1 for layer in layers]
    elif moe_position == 'woST':
        return [0] * length
    else:
        print("moe_position is None or invalid, should have a valid value")
        return None