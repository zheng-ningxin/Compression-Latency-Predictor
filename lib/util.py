# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
import torch
import torch.nn as nn
from .engine.onnxruntime import *
from sklearn.ensemble import RandomForestRegressor


_logger = logging.getLogger(__name__)


def measure_latency(model, dummy_input, cfg):
    """
    measure the latency for the model.
    """
    data_dir = cfg.get('data_dir', './data')
    os.makedirs(data_dir, exist_ok=True)
    repeat_times = cfg.get('repeat_times', 10)
    if cfg['engine'] == 'onnxruntime':
        onnx_path = os.path.join(data_dir, 'onnx.onnx')
        torch.onnx.export(model, dummy_input, onnx_path)
        if cfg['device'] == 'gpu':
            assert torch.cuda.is_available()
            latency = onnx_run_gpu(onnx_path, dummy_input, runtimes=repeat_times)
        elif cfg['device'] == 'cpu':
            latency = onnx_run_cpu(onnx_path, dummy_input, run_times=repeat_times)
        else:
            pass
    return latency


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_channel_list(model):
    """
    Return the channel numbers of the convolutional layers
    of the model.

    Parameter
    ---------
    model: nn.Module
        the target model to predict the latency.
    Returns
    -------
    channels: list
        the channel numbers of the conv layers in the model.
    """
    channels = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            channels.append(module.out_channels)
    return channels

def create_predictor(algo):
    if algo == 'randomforest':
        return RandomForestRegressor()
    return None