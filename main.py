import warnings 
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
sys.path.append('..')
import yaml 
import argparse
import traceback
import time
import torch
import os

from model.models import MoESTar
from model.trainer import Trainer
from lib.dataloader import get_dataloader_from_train_val_test, get_dataloader_from_index_data
from lib.utils import (
    init_seed,
    get_model_params,
    get_param_groups,
    load_graph, 
)
from model.utils.trans_utils import get_shortpath_num, get_num_degree

def model_supervisor(args):
    init_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = 'cpu'
    
    ## load dataset
    if args.dataset in ['METRLA', 'PEMSBAY']:
        DATA_LOADER = get_dataloader_from_index_data
    else: # ['NYCBike1', 'NYCBike2', 'NYCTaxi']
        DATA_LOADER = get_dataloader_from_train_val_test
    dataloader = DATA_LOADER(
        data_dir=args.data_dir, 
        dataset=args.dataset, 
        d_input=args.d_input,
        d_output=args.d_output,
        batch_size=args.batch_size, 
        test_batch_size=args.test_batch_size,
    )
    graph = load_graph(args.graph_file, device=args.device)
    assert args.num_nodes == len(graph), "num_nodes not right"
    args.num_shortpath, _ = get_shortpath_num(graph, args.dataset)
    args.num_node_deg = get_num_degree(graph)
    
    ## init model and set optimizer
    model = MoESTar(args).to(args.device)
    if args.moe_mlr == True:
        model_parameters = get_param_groups(model, args.lr_init, args.num_experts, args.top_k, args.moe_status)
        optimizer = torch.optim.Adam(params=model_parameters)
    else:
        model_parameters = get_model_params([model])
        optimizer = torch.optim.Adam(
            params=model_parameters, 
            lr=args.lr_init, 
            eps=1.0e-8, 
            weight_decay=0, 
            amsgrad=False
        )
    
    # 根据参数选择Learning Rate Scheduler
    if args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.factor, patience=args.patience)
    else:
        scheduler = None
        
    ## start training
    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler,
        dataloader=dataloader,
        graph=graph, 
        args=args
    )
    results = None
    try:
        if args.mode == 'train':
            results = trainer.train() # best_eval_loss, best_epoch
        elif args.mode == 'test':
            # test
            state_dict = torch.load(
                args.best_path,
                map_location=torch.device(args.device)
            )
            model.load_state_dict(state_dict['model'])
            print("Load saved model")
            results = trainer.test(model, dataloader['test'], dataloader['scaler'],
                        graph, trainer.logger, trainer.args)
        else:
            raise ValueError
    except:
        trainer.logger.info(traceback.format_exc())
    return results

if __name__=='__main__':
    # python main.py -g=$1 -d=$2 -s=$3
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='GPU ID to use')
    parser.add_argument('-d', '--dataset', default='NYCBike1', type=str, help='Dataset to use')
    parser.add_argument('-s', '--save_path', type=str, default=None, help='save path of log file')
    args = parser.parse_args()
    config_filename = f'configs/stgormer/{args.dataset}.yaml'
    
    print(f'Starting experiment with configurations in {config_filename}...')
    configs = yaml.load(
        open(config_filename), 
        Loader=yaml.FullLoader
    )
    configs['save_path'] = args.save_path
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    config_args = argparse.Namespace(**configs)
    model_supervisor(config_args)