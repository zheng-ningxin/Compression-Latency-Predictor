# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torchvision
from lib.latency_predictor import LatencyPredictor
net = torchvision.models.resnet101().cuda()

lp = LatencyPredictor(net, torch.zeros(16, 3, 224, 224).cuda())
lp.build('./config/example.yaml')