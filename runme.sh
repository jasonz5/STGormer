#!/bin/bash

# set the experiment GPU by the first parameters
# export CUDA_VISIBLE_DEVICES=$1
# set the dataset by the second parameters
python main_moe.py --gpu_id=$1 --config_filename=configs_moe/$2.yaml
