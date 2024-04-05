import torch.nn as nn

from lib.utils import masked_mae_loss

from model.layers import (
    STAttention, 
    MLP, 
)

class STAtt(nn.Module):
    def __init__(self, args):
        super(STAtt, self).__init__()
        # spatial temporal encoder
        # self.encoder = STAttention(Kt=3, Ks=3, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
        #                 input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout)
        self.encoder = STAttention(args.d_input, args.d_model, args.num_heads, args.mlp_ratio, args.encoder_depth, args.dropout)
        
        # traffic flow prediction branch
        self.mlp = MLP(args.d_model, args.d_output)
        self.mae = masked_mae_loss(mask_value=5.0)
        self.args = args
    
    def forward(self, view, graph):
        repr = self.encoder(view, graph) # view: n,l,v,c; graph: v,v 
        return repr  #nl(=1)vc

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
    

    