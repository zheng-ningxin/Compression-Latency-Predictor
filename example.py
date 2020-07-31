# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torchvision
from lib.latency_predictor import LatencyPredictor
net = torchvision.models.resnet18()

lp = LatencyPredictor(net, torch.zeros(1, 3, 224, 224))
lp.build('./config/example.yaml')