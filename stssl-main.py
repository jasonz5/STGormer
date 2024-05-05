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

from baselines.stssl.models import STSSL
from baselines.stssl.trainer import Trainer
from lib.dataloader import get_dataloader
from lib.utils import (
    init_seed,
    get_model_params,
    load_graph, 
    graph_unify, 
)

def model_supervisor(args):
    init_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = 'cpu'
    
    ## load dataset
    dataloader = get_dataloader(
        data_dir=args.data_dir, 
        dataset=args.dataset, 
        d_input=args.d_input,
        d_output=args.d_output,
        batch_size=args.batch_size, 
        test_batch_size=args.test_batch_size,
    )
    graph = load_graph(args.graph_file, device=args.device)
    if args.dataset in ['METRLA', 'PEMSBAY']:
        graph = graph_unify(graph)
    assert args.num_nodes == len(graph), "num_nodes not right"
    
    ## init model and set optimizer
    model = STSSL(args).to(args.device)
    model_parameters = get_model_params([model])
    optimizer = torch.optim.Adam(
        params=model_parameters, 
        lr=args.lr_init, 
        eps=1.0e-8, 
        weight_decay=0, 
        amsgrad=False
    )

    ## start training
    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_id', type=str, default='7', help='GPU ID to use')
    parser.add_argument('-d', '--dataset', default='NYCTaxi', type=str, help='Dataset to use')
    parser.add_argument('-s', '--save_path', type=str, default=None, help='save path of log file')
    args = parser.parse_args()
    config_filename = f'configs/stssl/{args.dataset}.yaml'
    
    print(f'Starting experiment with configurations in {config_filename}...')
    configs = yaml.load(
        open(config_filename), 
        Loader=yaml.FullLoader
    )
    configs['save_path'] = args.save_path
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    config_args = argparse.Namespace(**configs)
    model_supervisor(config_args)