# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import argparse
import torchvision
from lib.latency_predictor import LatencyPredictor

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model to measure the latency')
parser.add_argument('--config', help='the config file for the latency measurement')
args = parser.parse_args()
Model = getattr(torchvision.models, args.model)
net = Model(pretrained=True).cuda()

lp = LatencyPredictor(net, torch.ones(16, 3, 224, 224).cuda())
# lp.build('./config/example.yaml')
lp.generate_dataset(args.config)