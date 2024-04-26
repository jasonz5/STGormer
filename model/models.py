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
        # spatial temporal encoder
        # self.encoder = STAttention(Kt=3, Ks=3, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
        #                 input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout)
        args_moe = {"moe_status": args.moe_status, "num_experts": args.num_experts,
                    "moe_dropout": args.moe_dropout, "top_k": args.top_k, 
                    "moe_add_ff": args.moe_add_ff, 
                    "expertWeightsAda": args.expertWeightsAda, 'expertWeights': args.expertWeights}
        self.encoder = STAttention(
            args.d_input, args.d_model, args.num_heads, args.mlp_ratio,
            args.layer_depth, args.dropout, args.layers, args.sparseS, args.sparseT, 
            args_moe = args_moe, moe_position = args.moe_position)
        
        # traffic flow prediction branch
        self.mlp = MLP(args.d_model, args.d_output)
        self.mae = masked_mae_loss(mask_value=5.0)
        self.args = args
        # Filter 
        self.fft_status = args.fft_status
        self.filter = FilterLayer(args.num_nodes, args.input_length)
    
    def forward(self, view, graph):
        # Filter 
        if self.fft_status:
            # view: n,l,v,c -> n,c,v,l -> n,l,v,c
            view = view.permute(0, 3, 2, 1)
            view = self.filter(view)
            view = view.permute(0, 3, 2, 1)
        repr, aux_loss = self.encoder(view, graph) # view: n,l,v,c; graph: v,v 
        return repr, aux_loss  #nl(=1)vc

    def predict(self, repr):
        '''Predicting future traffic flow.
        :param repr (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        return self.mlp(repr)

    def loss(self, repr, y_true, scaler):
        loss = self.pred_loss(repr, y_true, scaler)
        return loss

    def pred_loss(self, repr, y_true, scaler):
        y_pred = scaler.inverse_transform(self.predict(repr))
        y_true = scaler.inverse_transform(y_true)
 
        loss = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0]) + \
                (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
        return loss
    