import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

import sys
from .positional_encoding import Positional1DEncoding
from .transformer_layers import TrandformerEncoder
from .utils.trans_utils import TemporalNodeFeature, SpatialNodeFeature, SpatialAttnBias, generate_moe_posList, get_degree_array

class STAttention(nn.Module):

    def __init__(
        self, in_channel, embed_dim, num_heads, mlp_ratio, layer_depth, 
        dropout, layers = None, args_attn = None,
        args_moe = None, moe_position = None, dataset = None):
        super(STAttention, self).__init__()
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.layer_depth = layer_depth
        self.layers = layers
        self.pos_embed_T = args_attn["pos_embed_T"]
        self.num_timestamps = args_attn["num_timestamps"]
        self.tod_scaler = args_attn["tod_scaler"]
        self.cen_embed_S = args_attn["cen_embed_S"]
        self.attn_mask_S = args_attn["attn_mask_S"]
        self.num_shortpath = args_attn["num_shortpath"]
        self.num_node_deg = args_attn["num_node_deg"]
        self.attn_bias_S = args_attn["attn_bias_S"]
        self.attn_mask_T = args_attn["attn_mask_T"]
        self.d_time_embed = args_attn["d_time_embed"]
        self.d_space_embed = args_attn["d_space_embed"]
        self.dataset = dataset
        
        # temporal encoding
        self.positional_encoding_1d = Positional1DEncoding()
        self.temporal_node_feature = TemporalNodeFeature(self.d_time_embed, self.num_timestamps, 
                                    scaler=self.tod_scaler, steps_per_day = args_attn["steps_per_day"])
        # spatial encoding
        self.spatial_node_feature = SpatialNodeFeature(self.num_node_deg, self.d_space_embed)
        input_embed_dim = embed_dim + int(self.pos_embed_T == "timestamp") * self.d_time_embed + int(self.cen_embed_S) * self.d_space_embed
        self.cat_st_embed = nn.Linear(input_embed_dim, embed_dim)
        self.spatial_attn_bias = SpatialAttnBias(self.num_shortpath)
        
        self.project = nn.Linear(in_channel, embed_dim)
        
        # 确定MoE化位置
        moe_posList = generate_moe_posList(layers, moe_position)
        
        args_wo_moe = args_moe.copy()
        args_wo_moe["moe_status"] = None
        self.st_encoder = nn.ModuleList([
            TrandformerEncoder(embed_dim, layer_depth, mlp_ratio, num_heads, dropout,\
                args_moe if moe_posList[i]==1 else args_wo_moe)
            for i in range(len(layers))])
        
    def forward(self, history_data, graph=None): # history_data: [B,T,N,C]; graph: [N,N] 
        ### fetch the tod/dow
        history_data = history_data.permute(0, 2, 1, 3) # [B, N, T, C]
        tod = history_data[..., -2]
        dow = history_data[..., -1]
        flow_data = history_data[..., :self.in_channel] # 不包含tod+dow
        # project the #dim of input to #embed_dim
        encoder_input = self.project(flow_data)  # B, N, T, D
        B, N, T, D = encoder_input.shape
        
        # temporal timestamps feature
        if  self.pos_embed_T == "timestamp": 
            time_feature =  self.temporal_node_feature(tod, dow) # [B, N, T] -> [B, N, T, D]
            encoder_input = torch.cat([encoder_input, time_feature], dim = -1)
        elif self.pos_embed_T == "timepos": # Transformer传统的1D位置编码
            encoder_input = encoder_input.contiguous().view(B*N, T, D)
            encoder_input, _ = self.positional_encoding_1d(encoder_input) # B*N, T, D
            encoder_input = encoder_input.view(B, N, T, D) # B, N, T, D

        # spatial node degree information
        if  self.cen_embed_S:
            degree = get_degree_array(graph) # [N]
            degree_feature = self.spatial_node_feature(degree)\
                .view(1,N,1,-1).expand(B, N, T, self.d_space_embed) # [N, D] -> [B, N, T, D]
            encoder_input = torch.cat([encoder_input, degree_feature], dim = -1)
        if  self.pos_embed_T == "timestamp" or self.cen_embed_S:
            encoder_input = self.cat_st_embed(encoder_input)

        ## 计算时间和空间的掩码矩阵 ##
        maskS, maskT = None, None
        if self.attn_mask_S:
            maskS = torch.ones((N, N)).to('cuda')
            torch.diagonal(maskS).fill_(0)
        if self.attn_mask_T:
            maskT = torch.tril(torch.ones((T, T))).to('cuda')
        
        ## 计算spatial的attn bias
        attn_bias_spatial = None
        if self.attn_bias_S:
            attn_bias_spatial = self.spatial_attn_bias(graph, self.dataset) # [n, n, 1]
        
        aux_loss = torch.tensor(0, dtype=torch.float32).to('cuda')
        layers_full = ['T'] + self.layers
        for i in range(1, len(layers_full)):
            if layers_full[i] != layers_full[i-1]:
                encoder_input = encoder_input.transpose(-2,-3)
            if layers_full[i] == 'S':
                encoder_input, loss, *_ = self.st_encoder[i-1](encoder_input, 
                        mask=maskS, attn_bias = attn_bias_spatial)
            else: # == 'T'
                encoder_input, loss, *_ = self.st_encoder[i-1](encoder_input, mask=maskT)
            aux_loss += loss
        if layers_full[-1] == 'S':
            encoder_input = encoder_input.transpose(-2,-3)
        # ipdb.set_trace()
        return encoder_input, aux_loss # # [B, N, T, D] [1]


########################################
## An MLP predictor
########################################
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(input_dim // 2)),
            nn.Tanh(),
            nn.Linear(int(input_dim // 2), output_dim)
        )
    def forward(self, x):
        x = self.network(x)
        return x
    
########################################
## Test Function
########################################
def main():
    import sys
    import os
    import torch
    from torchinfo import summary
    import contextlib
    
    # 设置CUDA设备
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = STAttention().to(device)

    # 定义日志文件的路径
    log_path = "../log/statt.log"  # 根据您的目录结构调整路径

    with open(log_path, "w") as log_file:
        with contextlib.redirect_stdout(log_file):
            summary(model, input_size=(1, 19, 128, 2), device=device)

def test():
    layers = ['T', 'S', 'T', 'T', 'S', 'T']
    moe_position = 'Half'
    moe_posList = generate_moe_posList(layers, moe_position)
    print(moe_posList)
    
    
if __name__ == '__main__':
    test()