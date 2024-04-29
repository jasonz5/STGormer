''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Modules import ScaledDotProductAttention
from ..moe.stmoe.st_moe_pytorch import MoE as STMoE, SparseMoEBlock
from ..moe.sharedmoe.mixture_of_experts import RoutedMoE
from ..moe.vanillamoe.mixture_of_experts import MoE
from ..moe.softmoe.softmoe import ContinuousMoE, SoftMoE


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
        

    def forward(self, q, k, v, mask=None, attn_bias = None):

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

        output, attn = self.attention(q, k, v, mask=mask, attn_bias=attn_bias)
        
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

class SoftMoEFNN(nn.Module):
    ''' Continuous MoE FNN module '''
    def __init__(self, input_dim, num_experts, hidden_dim=None, dropout=0.1):
        super(SoftMoEFNN, self).__init__()
        self.softExpert = SoftMoE(input_dim, num_experts) #SoftMoE ContinuousMoE
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x #[b,n,d]
        x, loss = self.softExpert(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x, loss
    
class SharedMoEFNN(nn.Module):
    ''' SharedMoE FNN module '''
    def __init__(self, input_dim, num_experts, hidden_dim=None, dropout=0.1, 
                 expertWeightsAda=False, expertWeights = None):
        super(SharedMoEFNN, self).__init__()
        self.sharedExpert = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.selectedExpert = RoutedMoE(input_dim, num_experts, hidden_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

        # shared/selected expert 权重分配
        self.expertWeightsAda = expertWeightsAda
        if self.expertWeightsAda:
            self.expertWeights = nn.Sequential(
                nn.Linear(input_dim, 2),
                nn.Softmax(dim=-1)
            )
        else:
            self.expertWeights = expertWeights

    def forward(self, x):
        residual = x #[b,n,d]
        x1 = self.sharedExpert(x)
        x2, loss = self.selectedExpert(x)
        
        # shared/selected expert 权重分配
        if self.expertWeightsAda:
            balance = self.expertWeights(x)  #[b,n,2]
            x = balance[..., 0:1]*x1 + balance[..., 1:2]*x2
        else:
            x = self.expertWeights[0]*x1 + self.expertWeights[1]*x2
        # import ipdb; ipdb.set_trace()
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x, loss
    
    
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

class STMoEFNN(nn.Module):
    ''' ST-MOE FNN module '''
    def __init__(self, d_in, num_experts, d_hid=None, dropout=0.1, moe_add_ff=False):
        super(STMoEFNN, self).__init__()
        self.moefnn = STMoE(
            dim = d_in,
            num_experts = num_experts,      # increase the experts (# parameters) of your model without increasing computation
            gating_top_n = 2,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
            threshold_train = 0.2,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
            threshold_eval = 0.2,
            capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            balance_loss_coef = 1e-2,       # multiplier on the auxiliary expert balancing auxiliary loss
            router_z_loss_coef = 1e-3,      # loss weight for router z-loss
        )
        self.moe_block = SparseMoEBlock(
            self.moefnn,
            add_ff_before = moe_add_ff,
            add_ff_after = moe_add_ff
        )
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input: (B, N, D) 
        # return: (B, N, D), (1,) (1,), (1,)
        x, total_aux_loss, balance_loss, router_z_loss = self.moe_block(x) 
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x, total_aux_loss.item()