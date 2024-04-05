import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from entmax import sparsemax, entmax15, entmax_bisect


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -np.inf)

        attn = F.softmax(attn, dim=-1)

        # entmax15 1.5倍稀疏softmax
        # attn = entmax15(attn)
        # sparsemax 2倍稀疏softmax
        #attn = sparsemax(attn)

        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn

def sparse_dot_product(key, value, query, k=0):
    scores = torch.matmul(query, key.transpose(2, 3))

    if (k > key.size()[1]):
        k = key.size()[1]

    if k:
        v, _ = torch.topk(scores, k)
        vk = v[:, :, -1].unsqueeze(2).expand_as(scores)
        mask_k = torch.lt(scores, vk)
        scores = scores.masked_fill(mask_k, -np.inf)

    attn = F.softamx(scores)
    context = torch.matmul(attn, value)
    return context, attn