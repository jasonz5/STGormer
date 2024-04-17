''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer.Modules import ScaledDotProductAttention
from model.moe.mixture_of_experts import MoE, HeirarchicalMoE, Experts

''' Adjust according to STGSP '''

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual
        output = self.layer_norm(output)
        return output, attn
    

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
    
class MoEFNN(nn.Module):
    ''' MOE FNN module '''

    def __init__(self, d_in, num_experts, d_hid=None, dropout=0.1):
        super(MoEFNN, self).__init__()
        self.moefnn = MoE(d_in, num_experts, d_hid)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x, loss = self.moefnn(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x, loss



# The MoE below is realized by chatting with ChatGPT.
# Current status: ignored
class ExpertFNN(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(ExpertFNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.network(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class ExpertFNNMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=6, dropout=0.1):
        super(ExpertFNNMoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([ExpertFNN(input_dim, hidden_dim, dropout=dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x, top_k=0):
        # 为每个时间步应用门控机制, 假设 x 的维度是 [batch_size, len_seq, input_dim]
        gating_distribution = self.gate(x)   # [batch_size, len_seq, num_experts]

        # 如果top_k大于num_experts，则重置top_k为num_experts
        if (top_k > gating_distribution.size()[-1]):
            top_k = gating_distribution.size()[-1]
        # 如果top_k非零，则专家稀疏化
        if top_k:
            v, _ = torch.topk(gating_distribution, top_k, dim=-1) 
            vk = v[:, :, :, -1].unsqueeze(-1).expand_as(gating_distribution)
            mask_k = torch.lt(gating_distribution, vk)
            gating_distribution = gating_distribution.masked_fill(mask_k, float('-inf'))
        gating_distribution = F.softmax(gating_distribution, dim=-1)
        
        
        # 获取每个专家的输出, #TODO:这一部分即使稀疏化后Expert也都参与了计算，并没有减少计算量
        expert_outputs = [expert(x) for expert in self.experts]  # List of [batch_size, len_seq, output_dim]
        expert_outputs = torch.stack(expert_outputs, dim=2) # [batch_size, len_seq, num_experts, output_dim]

        # 将专家的输出与相应的权重相乘，并求和
        output = torch.einsum('blnd,bln->bld', expert_outputs, gating_distribution)
        
        return output, gating_distribution