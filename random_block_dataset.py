# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import yaml
import json
import torch
import argparse
from lib.random_block import generate_model
from lib.util import measure_latency

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='The path of the config file')
    parser.add_argument('--outdir', default='./data', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    with open(args.config, 'r') as conf_f:
        config = yaml.safe_load(conf_f)

    for i in range(config['sample_count']):
        print('Sample %d random block models' % i)
        model = generate_model(config)
        dummy_input = torch.ones(1, 3, 224, 224)
        dummy_output = model(dummy_input)
        # print(model)
        module_level = config.get('submodule_level', 0)

        latency = measure_latency(model, dummy_input, config, module_level)
        file_name = 'random_block_%d.json'
        file_name = os.path.join(args.out_dir, file_name)
        result = [str(model), latency]
        with open(file_name, 'w') as jf:
            json.dump(result, jf)

if __name__ == '__main__':
    main()
