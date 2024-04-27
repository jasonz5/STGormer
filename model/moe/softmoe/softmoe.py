import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        x = self.network(x)
        return x
    
class ContinuousMoE(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=None, gating_hidden_dim=None, loss_coef = 1e-2):
        super(ContinuousMoE, self).__init__()
        hidden_dim = default(hidden_dim, input_dim*4)
        gating_hidden_dim = default(gating_hidden_dim, input_dim*2)
        
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gating_function = nn.Sequential(
            nn.Linear(input_dim, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, num_experts)
        )
        self.loss_coef = loss_coef

    def forward(self, x):
        expert_outputs = [F.relu(expert(x)) for expert in self.experts]  # List of [B, N, D]
        expert_outputs = torch.stack(expert_outputs, dim=2) # [B, N, E, D]

        gating_distribution = self.gating_function(x) # [B, N, E]
        gating_distribution = F.softmax(gating_distribution, dim=-1)

        output = torch.einsum('bned,bne->bnd', expert_outputs, gating_distribution)  
        
        mean_weights = gating_distribution.mean(dim=[0, 1]) 
        capacity_loss = torch.sum(mean_weights ** 2)
        return output, capacity_loss * self.loss_coef

    
class SoftMoE(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=None, gating_hidden_dim=None, loss_coef = 1e-2):
        super(SoftMoE, self).__init__()
        hidden_dim = default(hidden_dim, input_dim*4)
        gating_hidden_dim = default(gating_hidden_dim, input_dim*2)
        
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gating_function = nn.Sequential(
            nn.Linear(input_dim, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, num_experts)
        )
        self.loss_coef = loss_coef
        
    def forward(self, x, top_k=0):
        gating_distribution = self.gating_function(x)   # [B, N, E]

        if (top_k > gating_distribution.size()[-1]):
            top_k = gating_distribution.size()[-1]
        if top_k != 0:
            top_k_logits, indices = gating_distribution.topk(top_k, dim=-1) 
            zeros = torch.full_like(gating_distribution, float('-inf'))
            gating_distribution = zeros.scatter(-1, indices, top_k_logits)
        gating_distribution = F.softmax(gating_distribution, dim=-1)  # [B, N, E]
        
        expert_outputs = [expert(x) for expert in self.experts]  # List of [B, N, D]
        expert_outputs = torch.stack(expert_outputs, dim=2) # [B, N, E, D]

        output = torch.einsum('bned,bne->bnd', expert_outputs, gating_distribution)
        
        mean_weights = gating_distribution.mean(dim=[0, 1]) 
        capacity_loss = torch.sum(mean_weights ** 2)
        return output, capacity_loss * self.loss_coef