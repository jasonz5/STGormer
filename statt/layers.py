import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

from statt.positional_encoding import PositionalEncoding
from statt.transformer_layers import TransformerLayers


class STAttention(nn.Module):

    def __init__(self, in_channel, embed_dim, num_heads, mlp_ratio, encoder_depth, dropout):
        super(STAttention, self).__init__()
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.encoder_depth = encoder_depth

        # norm layers
        self.encoder_norm1 = nn.LayerNorm(embed_dim)
        self.encoder_norm2 = nn.LayerNorm(embed_dim)
        # positional encoding
        self.pos_mat=None
        self.positional_encoding = PositionalEncoding()
        
        # blocks
        self.project = FCLayer(in_channel, embed_dim)
        self.encoder_t11 = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        self.encoder_s12 = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        self.encoder_t13 = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        self.encoder_t21 = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        self.encoder_s22 = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        self.encoder_t23 = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

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
        
        ## ST block 1
        encoder_input = self.encoder_t11(encoder_input) # B, N, P, d
        encoder_input = encoder_input.transpose(-2,-3)
        encoder_input = self.encoder_s12(encoder_input) # B, P, N, d
        encoder_input = encoder_input.transpose(-2,-3)
        encoder_input = self.encoder_t13(encoder_input) # B, N, P, d
        encoder_input = self.encoder_norm1(encoder_input)#.view(batch_size, num_nodes, -1, self.embed_dim)# B, N, P, d
        
        ## ST block 2
        encoder_input = self.encoder_t21(encoder_input) # B, N, P, d
        encoder_input = encoder_input.transpose(-2,-3)
        encoder_input = self.encoder_s22(encoder_input) # B, P, N, d
        encoder_input = encoder_input.transpose(-2,-3)
        encoder_input = self.encoder_t23(encoder_input) # B, N, P, d
        encoder_input = self.encoder_norm2(encoder_input)#.view(batch_size, num_nodes, -1, self.embed_dim)# B, N, P, d
        return encoder_input



    def forward(self, history_data, graph): # history_data: n,l,v,c; graph: v,v 
        repr = self.encoding(history_data) # B, N, P, d
        repr = repr.transpose(-2,-3) # n,l,v,c
        return repr[:,-1:,:,:]



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
    from torchsummary import summary
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '2'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model = STAttention(
        patch_size=12,
        in_channel=1,
        embed_dim=96,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        mask_ratio=0.75,
        encoder_depth=4,
        decoder_depth=1,
        mode="pre-train"
    ).to(device)
    summary(model, (288*7, 307, 1), device=device)


if __name__ == '__main__':
    main()
