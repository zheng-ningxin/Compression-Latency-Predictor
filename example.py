# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torchvision
from lib.latency_predictor import LatencyPredictor
net = torchvision.models.resnet101()

lp = LatencyPredictor(net, torch.zeros(64, 3, 224, 224))
lp.build('./config/example.yaml')