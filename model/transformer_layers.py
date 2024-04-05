import math
from torch import nn
from model.transformer.Layers import EncoderLayer

class TransformerLayers(nn.Module):
    def __init__(self, d_model, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner=d_model*mlp_ratio, n_head=num_heads, d_k=d_model, d_v=d_model, dropout=dropout)
            for _ in range(nlayers)])

    def forward(self, src, mask=None, return_attns=False):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src=src.contiguous()
        src = src.view(B*N, L, D)
        
        enc_slf_attn_list = []
        enc_output = src
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        enc_output = enc_output.view(B, N, L, D)
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
    

# from torch.nn import TransformerEncoder, TransformerEncoderLayer
# class TransformerLayers(nn.Module):
#     def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.d_model = hidden_dim
#         encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

#     def forward(self, src):
#         B, N, L, D = src.shape
#         src = src * math.sqrt(self.d_model)
#         src=src.contiguous()
#         src = src.view(B*N, L, D)
#         src = src.transpose(0, 1)
#         output = self.transformer_encoder(src, mask=None)
#         output = output.transpose(0, 1).view(B, N, L, D)
#         return output
