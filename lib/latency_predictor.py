# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import yaml
import random
import logging
import torch
import torch.nn as nn
import torch.Tensor as Tensor
from nni._graph_utils import TorchModuleGraph
from nni.compression.torch.utils.shape_dependency import *
from .util import *

_logger = logging.getLogger(__name__)


class LatencyPredictor:
    def __init__(self, model, dummy_input):
        """
        Latency predictor that predicts the latency of a specific
        model in the compression scenario.

        Parameters
        ----------
        model: torch.nn.Module
            the model to build the compression latency predictor
        dummy_input: torch.Tensor
            the dummy input used to measure the latency.

        """
        self.bound_model = model
        self.dummy_input = dummy_input
        self.parse_model()
        # the latency predictor for this model
        self.predictor = None

    def parse_model(self):
        """
        parse the model and find the target
        Parameters
        ----------
        model: torch.nn.Module
            the target model to predict the latency.
        dummy_input:
            the example input tensor for the model.

        """
        with torch.onnx.set_training(model, False):
            # We need to trace the model in this way, else it will have problems
            traced = torch.jit.trace(model, dummy_input)
        self.channel_depen = ChannelDependency(traced_model=traced)
        self.group_depen = GroupDependency(traced_model=traced)
        self.graph = self.channel_depen.graph
        self.name2module = {}
        self.filter_count = {}
        for name, module in self.bound_model.named_modules():
            self.name2module[name] = module
            if isinstance(module, nn.Conv2d):
                self.filter_count[name] = module.out_channels

    def generate_model(self, channel_cfg):
        """
        generate the models according to the channel_cfg.
        The generated model has the same network architecture
        with self.bound_model, but the out_channels of each
        conv layers are configured according to the channel_cfg.

        Parameters
        ----------
        channel_cfg: dict
            A dict object that stores the number of the out_channels
            of each convolutional layers.
            For example, {'conv1' : 256, 'conv2':512 } 
        """

        def new_forward(ori_forward, out_channels):
            def forward(*args, **kwargs):
                _inputs = get_tensors_from(args)
                _inputs.extend(get_tensors_from(kwargs))
                # the input of nn.Conv2d only has one tensor
                assert len(_inputs) == 1
                module = args[0]
                module.input_shape = _inputs[0].size()
                out = ori_forward(*args, **kwargs)
                N_out, C_out, H_out, W_out = out.size()
                module.output_shape = (N_out, out_channels, H_out, W_out)
                return torch.zeros(module.output_shape)
            return forward
        self.ori_forwards = {}
        model = copy.deepcopy(self.bound_model)
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                return
            module = self.name2module[name]
            assert isinstance(module, nn.Conv2d)
            _forward = getattr(module, 'forward')
            self.ori_forwards[name] = _forward
            if name in channel_cfg:
                _new_forward = new_forward(_forward, channel_cfg[name])
            else:
                _new_forward = new_forward(_forward, module.out_channels)
            setattr(module, 'forward', _new_forward)
        try:
            model(self.dummy_input)
        except Exception as err:
            _logger.warn('The updated model may have shape conflicts')
            _logger.warn(err)
            _logger.warn(channel_cfg)
            # the model is not valid, it has shape conflict
            # under the
            return None
        # replace the original conv layers
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if module.in_channels == module.input_shape[1] and\
                    module.out_channels == module.output_shape[1]:
                # no need to replace this layer, only reset its
                # original forward function
                setattr(module, 'forward', self.ori_forwards[name])
                continue
            new_attrs = {
                'in_channels': module.input_shape[1], 'out_channels': module.output_shape[1]}
            _new_conv = new_conv2d(module, new_attributes=new_attrs)
            super_module, leaf_module = get_module_by_name(model, name)
            setattr(super_module, name.split('.')[-1], _new_conv)
        return model
        
    def generate_channel_cfg(self):
        channel_cfg = {}
        channel_d_sets = self.channel_depen.dependency_sets
        group_d_sets = self.group_depen.dependency_sets
        for _set in channel_d_sets:
            _max_group = max([group_d_sets[name]] for name in _set)
            # the filter count of all these layers should be
            # divisible by the _max_group
            tmp_name = next(iter(_set))
            f_num = self.filter_count[tmp_name]
            # TODO double check how to handle the banlance
            # between pruning and not pruning
            if random.uniform(0, 1) > 0.5:
                # there is 50% probability that we donnot
                # prune these layers
                continue
            else:
                sampled_count = int((random.uniform() * f_num) // _max_group)
                if sampled_count == 0:
                    sampled_count = _max_group
                for layer in _set:
                    channel_cfg[layer] = sampled_count
        return channel_cfg
                
            

    def build(self, cfgpath):
        """
        Sample `sample_count` models that have different
        channel numbers with the origanl model. Measure
        the latencies of these generated models. Build
        the latency predictor based on the channel count and
        the measured latency.

        Parameters
        ----------
        cfg: str
            path of the cfg file
        """
        # generate the models that has the same architecuture
        # with the original model but has the different numbers
        # of the channels
        assert os.path.exists(cfgpath)
        with open(cfgpath, 'r') as cfg_f:
            cfg = yaml.load(cfg_f)
        already_sampled = 0

        while already_sampled < cfg['sample_count']:
            channel_cfg = self.generate_channel_cfg()
            net = self.generate_model(channel_cfg)
            if net is None:
                # generated model is not legal
                continue
            already_sampled += 1
            latency = measure_latency(model, cfg)
                    

    def predict(self, model):
        if self.predictor is None:
            return None

    def load(self, ckpath):
        pass

    def export(self, savepath):
        pass
