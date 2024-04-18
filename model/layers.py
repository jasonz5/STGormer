import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

import sys
from model.positional_encoding import PositionalEncoding
from model.transformer_layers import TrandformerEncoder


class STAttention(nn.Module):

    def __init__(
        self, in_channel, embed_dim, num_heads, mlp_ratio, encoder_depth, dropout, num_blocks=2, nlayers=3, 
        args_moe = None, moe_position = None):
        super(STAttention, self).__init__()
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.encoder_depth = encoder_depth
        self.num_blocks = num_blocks
        self.nlayers = nlayers

        # positional encoding
        self.pos_mat=None
        self.positional_encoding = PositionalEncoding()
        #TODO:之后可以考虑加入时间和空间维度的位置编码
        
        self.project = FCLayer(in_channel, embed_dim)
        
        ## 这里构造ST-Transformer块，并决定在哪一层MoE化
        moe_posList = []
        if moe_position == 'Full':
            moe_posList = [1, 1, 1, 1, 1, 1]
        elif moe_position == 'Half':
            moe_posList = [0, 0, 0, 1, 1, 1]
        elif moe_position == 'woS':
            moe_posList = [1, 0, 1, 1, 0, 1]
        elif moe_position == 'woT':
            moe_posList = [0, 1, 0, 0, 1, 0]
        elif moe_position == 'woST':
            moe_posList = [0, 0, 0, 0, 0, 0]
        elif moe_position is None:
            print("moe_position is None, should have a value")

        args_wo_moe = args_moe.copy()
        args_wo_moe["moe_status"] = False
        self.st_encoder = nn.ModuleList([
            TrandformerEncoder(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout,\
                args_moe if moe_posList[i]==1 else args_wo_moe)
            for i in range(num_blocks*nlayers)])
        
        # blocks:(TST)*num_blocks
        # self.st_encoder = nn.ModuleList([
        #     TrandformerEncoder(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout, args_moe)
        #     for _ in range(num_blocks*nlayers)])


    def encoding(self, history_data):
        """
        Args: history_data (torch.Tensor): history flow data [B, P, N, d] # n,l,v,c
        Returns: torch.Tensor: hidden states
        """
        # project the #dim of input flow to #embed_dim
        patches = self.project(history_data.permute(0, 3, 1, 2)) # nlvc->nclv
        patches = patches.permute(0, 3, 2, 1)         # B, N, P, d
        batch_size, num_nodes, num_time, num_dim = patches.shape
        
        # positional embedding
        encoder_input, self.pos_mat = self.positional_encoding(patches)# B, N, P, d
        
        aux_loss = 0
        for i in range(0, len(self.st_encoder), self.nlayers):
            encoder_input, loss, *_ = self.st_encoder[i](encoder_input) # B, N, P, d
            aux_loss += loss
            encoder_input = encoder_input.transpose(-2,-3)
            encoder_input, loss, *_ = self.st_encoder[i+1](encoder_input) # B, P, N, d
            aux_loss += loss
            encoder_input = encoder_input.transpose(-2,-3)
            encoder_input, loss, *_ = self.st_encoder[i+2](encoder_input) # B, N, P, d
            aux_loss += loss
        '''
        ## ST block 1 (deprecated)
        encoder_input = self.encoder_t11(encoder_input) # B, N, P, d
        encoder_input = encoder_input.transpose(-2,-3)
        encoder_input = self.encoder_s12(encoder_input) # B, P, N, d
        encoder_input = encoder_input.transpose(-2,-3)
        encoder_input = self.encoder_t13(encoder_input) # B, N, P, d
        '''
        
        return encoder_input, aux_loss



    def forward(self, history_data, graph=None): # history_data: n,l,v,c; graph: v,v 
        repr, aux_loss = self.encoding(history_data) # B, N, P, d
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
    
    model = STAttention(in_channel=2, embed_dim=64, num_heads=4, mlp_ratio=4, encoder_depth=1, dropout=0.1).to(device)

    # 定义日志文件的路径
    log_path = "../log/statt.log"  # 根据您的目录结构调整路径

    with open(log_path, "w") as log_file:
        with contextlib.redirect_stdout(log_file):
            summary(model, input_size=(1, 19, 128, 2), device=device)

if __name__ == '__main__':
    main()
