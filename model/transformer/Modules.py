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

    def sparse_dot_product(self, query, key, value, k=0):
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.temperature

        # 如果k大于序列长度，则重置k为序列长度
        if (k > key.size()[2]):
            k = key.size()[2]

        # 如果k非零，则执行稀疏操作
        if k:
            # v包含scores中最高的k个注意力weights。#dim : (batch_size, num_heads, seq_length_query, k)
            v, _ = torch.topk(scores, k, dim=-1) 
            # 首先选择每个query对应的第k大的score，广播到scores相同的维度大小
            vk = v[:, :, :, -1].unsqueeze(-1).expand_as(scores)
            # 所有小于其对应query的第k大score的元素位置为True，其余为False。
            mask_k = torch.lt(scores, vk)
            scores = scores.masked_fill(mask_k, float('-inf'))

        attn = F.softamx(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn