import sys
import os
import torch
import torch.nn as nn
from torchinfo import summary
import contextlib
import argparse
import time
import yaml 
from lib.utils import (
    init_seed,
    get_model_params,
    load_graph, 
)

from model.models import STSSL
from statt.layers import STAttention

def main(args):
    ## 设定要输出的模型
    stssl = STSSL(args).to(args.device)
    model_parameters = get_model_params([stssl])
    # 计算参数量和总大小
    total_params = sum(p.numel() for p in model_parameters)
    total_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes for float32
    log_path = "log/para.log"
    with open(log_path, "w") as log_file:
        log_file.write(f"ST-SSL\n")
        log_file.write(f"Total params: {total_params}\n")
        log_file.write(f"Params size (MB): {total_size_mb}\n")

    ## 设定要输出的模型
    statt = STAttention(in_channel=2, embed_dim=64, num_heads=4, mlp_ratio=4, encoder_depth=1, dropout=0.1).to(args.device)
    model_parameters = get_model_params([statt])
    # 计算参数量和总大小
    total_params = sum(p.numel() for p in model_parameters)
    total_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes for float32
    with open(log_path, "a") as log_file:
        log_file.write(f"ST-Attention\n")
        log_file.write(f"Total params: {total_params}\n")
        log_file.write(f"Params size (MB): {total_size_mb}\n")


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='7', help='GPU ID to use')
    parser.add_argument('--config_filename', default='configs/NYCBike1.yaml', 
                    type=str, help='the configuration to use')
    args = parser.parse_args()
    
    print(f'Starting experiment with configurations in {args.config_filename}...')
    time.sleep(1)
    configs = yaml.load(
        open(args.config_filename), 
        Loader=yaml.FullLoader
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args = argparse.Namespace(**configs)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    main(args)
