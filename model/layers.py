import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

import sys
from .positional_encoding import PositionalEncoding
from .transformer_layers import TrandformerEncoder
from .utils.trans_utils import SpatialNodeFeature, SpatialAttnBias, generate_moe_posList

class STAttention(nn.Module):

    def __init__(
        self, in_channel, embed_dim, num_heads, mlp_ratio, layer_depth, 
        dropout, layers = None, args_attn = None,
        args_moe = None, moe_position = None):
        super(STAttention, self).__init__()
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.layer_depth = layer_depth
        self.layers = layers
        self.attn_mask_S = args_attn["attn_mask_S"]
        self.attn_mask_T = args_attn["attn_mask_T"]
        self.attn_bias_S = args_attn["attn_bias_S"]
        self.attn_bias_T = args_attn["attn_bias_T"]
        self.pos_embed_T = args_attn["pos_embed_T"]
        self.cen_embed_S = args_attn["cen_embed_S"]
        self.num_spatial = args_attn["num_spatial"]
        self.num_degree = args_attn["num_degree"]
        
        # temporal encoding
        self.pos_mat=None
        self.positional_encoding = PositionalEncoding()
        # spatial encoding
        self.spatial_node_feature = SpatialNodeFeature(self.num_degree, self.embed_dim)
        self.spatial_attn_bias = SpatialAttnBias(self.num_spatial, bias_dim=1)
        
        self.project = FCLayer(in_channel, embed_dim)
        
        # 确定MoE化位置
        moe_posList = generate_moe_posList(layers, moe_position)
        
        args_wo_moe = args_moe.copy()
        args_wo_moe["moe_status"] = None
        self.st_encoder = nn.ModuleList([
            TrandformerEncoder(embed_dim, layer_depth, mlp_ratio, num_heads, dropout,\
                args_moe if moe_posList[i]==1 else args_wo_moe)
            for i in range(len(layers))])

    def encoding(self, history_data, graph):
        """
        Args: history_data (torch.Tensor): history flow data [B, P, N, d] # n,l,v,c
        Returns: torch.Tensor: hidden states
        """
        # project the #dim of input flow to #embed_dim
        patches = self.project(history_data.permute(0, 3, 1, 2)) # nlvc->nclv
        encoder_input = patches.permute(0, 3, 2, 1)         # B, N, P, d
        batch_size, num_nodes, num_time, num_dim = encoder_input.shape
        
        # temporal positional embedding
        if  self.pos_embed_T:
            encoder_input, self.pos_mat = self.positional_encoding(encoder_input)# B, N, P, d
        
        ## 计算时间和空间的掩码矩阵 ##
        if self.attn_mask_S:
            # maskS = graph.to('cuda')
            maskS = torch.ones((num_nodes, num_nodes)).to('cuda')
            torch.diagonal(maskS).fill_(0)
        else:
            maskS = None
        if self.attn_mask_T:
            maskT = torch.tril(torch.ones((num_time, num_time))).to('cuda')
        else:
            maskT = None
        
        # add spatial node degree information
        if self.cen_embed_S:
            degree = graph.sum(dim=-1).long()
            degree_feature = self.spatial_node_feature(degree) # [n, d]
            degree_feature = degree_feature.view(1, num_nodes, 1, num_dim)
            encoder_input = encoder_input + degree_feature # [b, n, t, d] [1, n, 1, d]
        ## 计算spatial的attn bias
        if self.attn_bias_S:
            attn_bias_spatial = self.spatial_attn_bias(graph) # [n, n, 1]
        else: 
            attn_bias_spatial = None
        if self.attn_bias_T:
            attn_bias_temporal = None
        else: 
            attn_bias_temporal = None
            
        aux_loss = torch.tensor(0, dtype=torch.float32).to('cuda')
        layers_full = ['T'] + self.layers
        for i in range(1, len(layers_full)):
            if layers_full[i] != layers_full[i-1]:
                encoder_input = encoder_input.transpose(-2,-3)
            if layers_full[i] == 'S':
                encoder_input, loss, *_ = self.st_encoder[i-1](encoder_input, 
                        mask=maskS, attn_bias = attn_bias_spatial)
            else: # == 'T'
                encoder_input, loss, *_ = self.st_encoder[i-1](encoder_input, 
                        mask=maskT, attn_bias = attn_bias_temporal)
            aux_loss += loss
        if layers_full[-1] == 'S':
            encoder_input = encoder_input.transpose(-2,-3)
        return encoder_input, aux_loss

    def forward(self, history_data, graph=None): # history_data: n,l,v,c; graph: v,v 
        repr, aux_loss = self.encoding(history_data, graph) # B, N, P, d
        repr = repr.transpose(-2,-3) # n,l,v,c
        return repr[:,-1:,:,:], aux_loss


########################################
## An MLP predictor
########################################
class MLP(nn.Module):
    def __init__(self, c_in, c_out): 
        super(MLP, self).__init__()
        self.fc1 = FCLayer(c_in, int(c_in // 2))
        self.fc2 = FCLayer(int(c_in // 2), c_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x.permute(0, 3, 1, 2))) # nlvc->nclv
        x = self.fc2(x).permute(0, 2, 3, 1) # nclv->nlvc
        return x

class FCLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCLayer, self).__init__()
        self.linear = nn.Conv2d(c_in, c_out, 1)  

    def forward(self, x):
        return self.linear(x)
    
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
    
    model = STAttention(in_channel=2, embed_dim=64, num_heads=4, mlp_ratio=4, layer_depth=1, dropout=0.1).to(device)

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