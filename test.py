#!/usr/bin/python3

import argparse
import torch.nn.functional as F
import numpy as np
from trainer import Cyc_Trainer
import yaml


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/CycleGan.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'CycleGan':
        trainer = Cyc_Trainer(config)

    # trainer.test()
    trainer.testBest()
    

###################################
if __name__ == '__main__':
    main()