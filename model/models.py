import torch.nn as nn

from lib.utils import masked_mae_loss

from .layers import (
    STAttention, 
    MLP, 
)
from .utils.filter import FilterLayer

class MoESTar(nn.Module):
    def __init__(self, args):
        super(MoESTar, self).__init__()
        args_moe = {"moe_status": args.moe_status, "num_experts": args.num_experts,
                    "moe_dropout": args.moe_dropout, "top_k": args.top_k, 
                    "moe_add_ff": args.moe_add_ff, 
                    "expertWeightsAda": args.expertWeightsAda, 'expertWeights': args.expertWeights}
        args_attn = {"pos_embed_T": args.pos_embed_T, "num_timestamps": args.num_timestamps, 
                    "tod_scaler": args.tod_scaler, "steps_per_day": args.steps_per_day,
                    "cen_embed_S": args.cen_embed_S, "attn_bias_S": args.attn_bias_S,
                    "num_shortpath": args.num_shortpath, "num_node_deg": args.num_node_deg,
                    "attn_mask_S": args.attn_mask_S, "attn_mask_T": args.attn_mask_T,
                    "d_time_embed": args.d_time_embed, "d_space_embed": args.d_space_embed}
        self.encoder = STAttention( #IMPORT: d_output
            args.d_output, args.d_model, args.num_heads, args.mlp_ratio,
            args.layer_depth, args.dropout, args.layers, args_attn = args_attn, 
            args_moe = args_moe, moe_position = args.moe_position, dataset = args.dataset)
        
        # traffic flow prediction branch
        if args.dataset in ['METRLA', 'PEMSBAY']:
            self.output_proj = MLP(args.input_length*args.d_model, args.output_length*args.d_output)
        else:
            self.output_proj = MLP(args.d_model, args.output_length*args.d_output)    
        self.mae = masked_mae_loss(mask_value=args.mask_value_train)
        self.args = args
        # Filter 
        self.fft_status = args.fft_status
        self.filter = FilterLayer(args.num_nodes, args.input_length)
    
    def forward(self, view, graph):
        # view: b,t,n,c; graph: n,n 
        # Filter 
        if self.fft_status:
            # view: b,t,n,d -> b,d,n,t -> b,t,n,d
            view = view.permute(0, 3, 2, 1)
            view = self.filter(view)
            view = view.permute(0, 3, 2, 1)
        repr, aux_loss = self.encoder(view, graph)  #[B, N, T, D]
        return repr, aux_loss  

    def predict(self, repr):
        '''Predicting future traffic flow.
        repr  # [B, N, T, D]
        '''
        if self.args.dataset in ['METRLA', 'PEMSBAY']:
            B, N, T, D = repr.shape
            out = repr.view(B, N, T * D)
            out = self.output_proj(out).view(B, N, T, 1)
            out = out.transpose(1, 2)  # [B, T, N, 1]
        else:
            out = self.output_proj(repr[:,:,-1:,:])
            out = out.permute(0,2,1,3)
        return out # [B, T, N, C]

    def loss(self, repr, y_true, scaler):
        y_pred = scaler.inverse_transform(self.predict(repr))
        y_true = scaler.inverse_transform(y_true)

        loss = self.mae(y_pred[..., :self.args.d_output], y_true[..., :self.args.d_output])
        return loss
    