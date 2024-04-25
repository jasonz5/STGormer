import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.fft


class FilterLayer(nn.Module):
    def __init__(self,
                 num_nodes=207,
                 seq_length=12,
                 residual_channels=32,
                 dilation_channels=32,
                 kernel_size=2,
                 dropout=0):
        self.seq_length = seq_length
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(
                1, num_nodes, seq_length // 2 + 1, 2, dtype=torch.float32) *
            (1 / seq_length))
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        batch, channel, nodes, seq_len = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=-1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=-1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        return hidden_states + input_tensor