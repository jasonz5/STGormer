import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
import scipy.sparse.csgraph as csgraph

class TemporalNodeFeature(nn.Module):
    def __init__(self, num_timestamps, d_model):
        super(TemporalNodeFeature, self).__init__()
        self.temporal_encoder = nn.Embedding(num_timestamps, d_model)

    def forward(self, timestamps):
        # timestamps: [t]  
        temporal_feature = self.temporal_encoder(timestamps) # [t, d]
        return temporal_feature

class SpatialNodeFeature(nn.Module):
    def __init__(self, num_degree, d_model):
        super(SpatialNodeFeature, self).__init__()
        self.degree_encoder = nn.Embedding(num_degree, d_model)

    def forward(self, degree):
        # degree: [n]  
        degree_feature = self.degree_encoder(degree) # [n, d]
        return degree_feature

class SpatialAttnBias(nn.Module):
    def __init__(self, num_spatial, bias_dim=1):
        super(SpatialAttnBias, self).__init__()
        self.bias_dim = bias_dim
        self.num_spatial = num_spatial
        self.spatial_pos_encoder = nn.Embedding(num_spatial, bias_dim)
        
    def forward(self, graph):
        # 使用 Floyd-Warshall 算法计算所有节点对的最短路径
        graph = modify_graph(graph)
        if graph.is_cuda:
            graph = graph.cpu().numpy()
        else:
            graph = graph.numpy()
        shortest_path = csgraph.floyd_warshall(graph, return_predecessors=False)
        shortest_path = torch.tensor(shortest_path, dtype=torch.long, device='cuda')
        num_spatial = int(torch.max(shortest_path)) + 1
        assert self.num_spatial == num_spatial, "num_spatial结果不相等"
        
        spatial_pos_bias = self.spatial_pos_encoder(shortest_path)
        return spatial_pos_bias

def get_num_degree(graph):
    degree = graph.sum(dim=-1)
    num_degree = int(torch.max(degree)) + 1
    return num_degree

def get_shortpath_num(graph):
    graph = modify_graph(graph)
    if graph.is_cuda:
        graph = graph.cpu().numpy()
    else:
        graph = graph.numpy()
    shortest_path = csgraph.floyd_warshall(graph, return_predecessors=False)
    shortest_path = torch.tensor(shortest_path, dtype=torch.long)
    num_spatial = int(torch.max(shortest_path)) + 1
    return num_spatial
    

# Replace 0s with float('inf'), except on the diagonal
def modify_graph(graph):
    modified_graph = graph.clone().float()
    inf_mask = (graph == 0) & ~torch.eye(graph.size(0), dtype=torch.bool, device=graph.device)
    modified_graph[inf_mask] = float('inf')
    # torch.diagonal(modified_graph)[:] = 0
    return modified_graph


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