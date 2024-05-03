import torch
from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer

class Positional1DEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        # input_data (torch.tensor): input sequence with shape [B, T, D].
        batch_size, num_times, num_feat = input_data.shape
        tp_enc_1d = PositionalEncoding1D(num_feat)
        pos_enc = tp_enc_1d(input_data)
        input_data += pos_enc
        return input_data, pos_enc

class Positional2DEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        # input_data (torch.tensor): input sequence with shape [B, N, T, D].
        batch_size, num_nodes, num_times, num_feat = input_data.shape
        tp_enc_2d = PositionalEncoding2D(num_feat)
        pos_enc = tp_enc_2d(input_data)
        input_data += pos_enc
        return input_data, pos_enc