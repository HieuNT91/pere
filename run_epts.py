import os
import argparse
from epts.epts_config import EptConfig
from epts.ept_phase1 import run_ept_phase1
from epts.ept_phase2 import run_ept_phase2
from epts.ept_dpe import run_ept_dpe
from epts.ept_maxvol import run_ept_maxvol
from epts.ept_phase2_vary_mode import run_ept_phase2_vary_mode

from utils.helpers import make_logger
import logging
import matplotlib
import numpy as np
import random

logging.getLogger('matplotlib').setLevel(logging.ERROR)

# all_datasets = ['synthetic_data', 'adult_data']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run experiments')
    parser.add_argument('--run-id', '-rid', default='0', type=str)
    parser.add_argument('--ept', '-e', dest='epts',
                        action='append', required=True)
    parser.add_argument('--datasets', dest='datasets', nargs='*')
    parser.add_argument('--methods', dest='methods', nargs='*')
    parser.add_argument('--num-proc', '-np', default=4, type=int)
    parser.add_argument('--update-config', '-uc', action='store_true')
    parser.add_argument('--seed', '-s', default=124, type=int)
    parser.add_argument('--start-seed', dest='start_seed', type=int, default=0)
    parser.add_argument('--end-seed', dest='end_seed', type=int, default=2)

    args = parser.parse_args()

    save_dir = f'results/run_{args.run_id}'
    config_path = os.path.join(save_dir, 'config.yml')

    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(args.seed - 1)
    random.seed(args.seed - 3)
    np.set_printoptions(suppress=True)

    ec = EptConfig()
    
    # update config if needed
    if args.update_config or not os.path.isfile(config_path):
        ec.to_file(config_path, mode='merge_cls')
    else:
        ec.from_file(config_path)

    for ept in set(args.epts):
        ept_dir = os.path.join(save_dir, f'ept_{ept}')
        os.makedirs(ept_dir, exist_ok=True)
        logger = make_logger(f'ept_{ept}', ept_dir)

        if ept == "phase1":
            run_ept_phase1(ec, ept_dir, datasets=args.datasets,
                        methods=args.methods, num_proc=args.num_proc,
                        start_seed=args.start_seed, end_seed=args.end_seed, logger=logger)
        
        elif ept == "phase2":
            run_ept_phase2(ec, ept_dir, datasets=args.datasets,
                        methods=args.methods, num_proc=args.num_proc,
                        start_seed=args.start_seed, end_seed=args.end_seed, logger=logger)
            
        elif ept == "dpe":
            run_ept_dpe(ec, ept_dir, datasets=args.datasets,
                        num_proc=args.num_proc,
                        start_seed=args.start_seed, end_seed=args.end_seed, logger=logger)
        
        elif ept == "maxvol":
            run_ept_maxvol(ec, ept_dir, datasets=args.datasets,
                        num_proc=args.num_proc,
                        start_seed=args.start_seed, end_seed=args.end_seed, logger=logger)
        
        elif ept == "vary_mode":
            run_ept_phase2_vary_mode(ec, ept_dir, datasets=args.datasets,
                        methods=args.methods, num_proc=args.num_proc,
                        start_seed=args.start_seed, end_seed=args.end_seed, logger=logger)
            

