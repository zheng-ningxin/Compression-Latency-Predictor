# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
import torch
import torch.nn as nn
from .engine.onnxruntime import *
from sklearn.ensemble import RandomForestRegressor


_logger = logging.getLogger(__name__)


def measure_module_latency(model, dummy_input, cfg):
    """
    measure the latency for the model.
    """
    data_dir = cfg.get('data_dir', './data')
    os.makedirs(data_dir, exist_ok=True)
    repeat_times = cfg.get('repeat_times', 10)
    if cfg['engine'] == 'onnxruntime':
        onnx_path = os.path.join(data_dir, 'onnx.onnx')
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        torch.onnx.export(model, dummy_input, onnx_path)
        if cfg['device'] == 'gpu':
            assert torch.cuda.is_available()
            latency = onnx_run_gpu(
                onnx_path, dummy_input, runtimes=repeat_times)
        elif cfg['device'] == 'cpu':
            latency = onnx_run_cpu(
                onnx_path, dummy_input, run_times=repeat_times)
        else:
            pass
    return latency


def dummy_input_forward_hook(inputs_dict, name):
    def forward_hook(module, inputs, output):
        # save the dummy_input for the module
        errmsg = 'currently only can measure the model with\
            one input tensor,{} has {} inputs'.format(module, len(inputs))
        assert(len(inputs) == 1), errmsg
        inputs_dict[name] = inputs[0]

    return forward_hook


def measure_latency(model, dummy_input, cfg, level=0):
    """
    measure the latency for the target model and its submodules.
    Parameters
    ----------
    model: torch.nn.Module
        The target model to measure the latency
    dummy_input: torch.tensor
        The dummy input for model to measure the latency. Note that,
        the dummy input should be on the same device with the model.
    cfg: dict
        Configure of the latency measurement environment, for example,
        the inference engine and the device(CPU/GPU).
    level: int
        If level is larger than 0, we will also measure the latencies of
        submodules whose level is less than `level`. For example, the level
        of layer1.0.conv1 is 3.
    """
    latencies = {}
    # measure the latency of the whole model first
    model_latency = measure_module_latency(model, dummy_input, cfg)
    latencies['model'] = model_latency
    if level == 0:
        return latencies
    module_dummy_inputs = {}

    hooks = []
    # register the forward hook for all the modules
    for name, module in model.named_modules():
        hook_handle = module.register_forward_hook(
            dummy_input_forward_hook(module_dummy_inputs, name))
        hooks.append(hook_handle)
    # forward inference
    model(dummy_input)
    for hook_handle in hooks:
        hook_handle.remove()
    # measure the latency for the submodules whose level is smaller
    # than `level`
    for name, module in model.named_modules():
        tmp = name.split('.')
        if len(tmp) > level:
            continue
        module_latency = measure_module_latency(module, module_dummy_inputs[name], cfg)
        latencies[name] = module_latency
    return latencies

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
