''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer

class TrandformerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src, mask=None, return_attns=False):
        enc_slf_attn_list = []
        enc_output = src

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

