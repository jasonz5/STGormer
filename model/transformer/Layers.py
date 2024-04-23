''' Define the Layers '''
import torch.nn as nn
from model.transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, SharedMoEFNN, MoEFNN , STMoEFNN


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(
        self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, 
        moe_status=False, num_experts=6, moe_dropout=0.1, top_k = 2,
        moe_add_ff = None, expertWeightsAda = False, expertWeights = None):
        super(EncoderLayer, self).__init__()
        self.moe_status = moe_status
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        if moe_status == 'SharedMoE':
            self.pos_ffn = SharedMoEFNN(d_model, num_experts, hidden_dim=d_model*4, dropout=moe_dropout, 
                                        expertWeightsAda=expertWeightsAda, expertWeights=expertWeights)
        elif moe_status == 'MoE':
            self.pos_ffn = MoEFNN(d_model, d_inner, dropout=dropout)
        elif moe_status == 'STMoE':
            self.pos_ffn = STMoEFNN(d_model, num_experts, dropout=moe_dropout, moe_add_ff=moe_add_ff)
        else: # None
            self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        aux_loss = 0
        if self.moe_status == 'SharedMoE':
            enc_output, aux_loss = self.pos_ffn(enc_output)
        elif self.moe_status == 'MoE':
            enc_output, aux_loss = self.pos_ffn(enc_output)
        elif self.moe_status == 'STMoE':
            enc_output, aux_loss = self.pos_ffn(enc_output)
        else: # None
            enc_output = self.pos_ffn(enc_output) 
        return enc_output, enc_slf_attn, aux_loss

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
