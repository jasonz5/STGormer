0. 默认的时空Attention Layer(不用管)
NYCBike1: layer3
NYCBike2: layer1
NYCTaxi: layer3
METRLA: layer3
PEMSBAY: layer1

1. python main.py -g={指定GPU} /
-d={指定数据集，NYCBike1/NYCBike2/NYCTaxi/METRLA/PEMSBAY} /
-s={自定义保存文件夹命名}

2. 消融实验更改：
w/o attn_bias_S: attn_bias_S项改为False，模型默认为True
w/o pos_embed_T: pos_embed_T项改为None，NYCBike12/NYCTaxi默认为timestamp，METRLA/PEMSBAY默认为timepos
w/o cen_embed_S: cen_embed_S项改为False，模型默认为True
w/o STMoE: moe_position项改为woST，模型默认为Full