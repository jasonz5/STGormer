import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

import sys
from .positional_encoding import Positional1DEncoding
from .transformer_layers import TrandformerEncoder
from .utils.trans_utils import TemporalNodeFeature, SpatialNodeFeature, SpatialAttnBias, generate_moe_posList

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
        self.pos_embed_T = args_attn["pos_embed_T"]
        self.num_timestamps = args_attn["num_timestamps"]
        self.tod_scaler = args_attn["tod_scaler"]
        self.cen_embed_S = args_attn["cen_embed_S"]
        self.attn_mask_S = args_attn["attn_mask_S"]
        self.num_shortpath = args_attn["num_shortpath"]
        self.num_node_deg = args_attn["num_node_deg"]
        self.attn_bias_S = args_attn["attn_bias_S"]
        self.attn_mask_T = args_attn["attn_mask_T"]
        
        # temporal encoding
        self.positional_encoding_1d = Positional1DEncoding()
        self.temporal_node_feature = TemporalNodeFeature(self.embed_dim, self.num_timestamps, scaler=self.tod_scaler)
        # spatial encoding
        self.spatial_node_feature = SpatialNodeFeature(self.num_node_deg, self.embed_dim)
        self.cat_st_embed = nn.Linear(embed_dim * 3, embed_dim)
        self.spatial_attn_bias = SpatialAttnBias(self.num_shortpath, bias_dim=1)
        
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
        Args: history_data (torch.Tensor): history flow data [B, T, N, c]
        Returns: torch.Tensor: hidden states
        """
        ### fetch the tod/dow
        history_data = history_data.permute(0, 2, 1, 3 ) # [B, N, T, c]
        tod = history_data[..., -2]
        dow = history_data[..., -1]
        history_data = history_data[..., :self.in_channel]
        # project the #dim of input flow to #embed_dim
        patches = self.project(history_data.permute(0, 3, 2, 1)) # [B, N, T, C]->[B,D,T,N]
        encoder_input = patches.permute(0, 3, 2, 1)         # B, N, T, D
        B, N, T, D = encoder_input.shape
        
        if  self.pos_embed_T and self.cen_embed_S:
            # temporal timestamps feature
            time_feature =  self.temporal_node_feature(tod, dow) # [B, N, T] -> [B, N, T, D]
            # spatial node degree information
            degree = graph.sum(dim=-1).long()
            degree_feature = self.spatial_node_feature(degree) # [N, D]
            # concatenation
            # import ipdb; ipdb.set_trace()
            encoder_input = self.cat_st_embed(
                torch.cat([encoder_input, time_feature, degree_feature.view(1,N,1,D).expand(B, N, T, D)], dim = -1))
        else: # Transformer传统的1D位置编码
            encoder_input = encoder_input.contiguous().view(B*N, T, D)
            encoder_input, _ = self.positional_encoding_1d(encoder_input) # B*N, T, D
            encoder_input = encoder_input.view(B, N, T, D) # B, N, T, D
            
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
            attn_bias_spatial = self.spatial_attn_bias(graph) # [n, n, 1]
            
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
        return encoder_input, aux_loss

    def forward(self, history_data, graph=None): # history_data: [B,T,N,D]; graph: [N,N] 
        repr, aux_loss = self.encoding(history_data, graph) # [B, N, T, D]
        repr = repr.transpose(-2,-3) # [B, T, N, D]
        return repr, aux_loss


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