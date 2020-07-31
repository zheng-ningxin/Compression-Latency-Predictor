# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import yaml
import random
import logging
import torch
import torch.nn as nn
from torch import Tensor
from nni._graph_utils import TorchModuleGraph
from nni.compression.torch.utils.shape_dependency import ChannelDependency, GroupDependency
from .util import *

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


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
        with torch.onnx.set_training(self.bound_model, False):
            # We need to trace the model in this way, else it will have problems
            traced = torch.jit.trace(self.bound_model, self.dummy_input)
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
        print('In genrate')

        def new_forward(ori_forward, new_size):
            def forward(*args, **kwargs):
                _inputs = get_tensors_from(args)
                _inputs.extend(get_tensors_from(kwargs))
                # the input of nn.Conv2d only has one tensor
                assert len(_inputs) == 1
                module = args[0]
                module.input_shape = _inputs[0].size()
                out = ori_forward(*args, **kwargs)
                output_shape = list(out.size())
                # modify the shape of the output tensor according to
                # the new_size
                for dim, _size in new_size:
                    output_shape[dim] = _size
                module.output_shape = output_shape
                # torch.zeros can use the list to specify the shape
                return torch.zeros(module.output_shape)
            return forward
        self.ori_forwards = {}
        print('test point1')
        model = copy.deepcopy(self.bound_model)
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d) and \
                    not isinstance(module, nn.Linear):
                continue
            module = self.name2module[name]
            # TODO notice the Linear Layer
            _forward = getattr(module, 'forward')
            self.ori_forwards[name] = _forward
            if name in channel_cfg:
                # The structure pruning of
                # Conv2D and Linear are both at the dimension-1
                _new_forward = new_forward(_forward, {1: channel_cfg[name]})
            else:
                _count = module.out_channels if isinstance(
                    module, nn.Conv2d) else module.out_features
                _new_forward = new_forward(_forward, {1: _count})
            setattr(module, 'forward', _new_forward)
        try:
            print('$$$$$$$')
            model(self.dummy_input)
        except Exception as err:
            _logger.warn('The updated model may have shape conflicts')
            _logger.warn(err)
            print(err)
            print('###########')
            _logger.warn(channel_cfg)
            # the model is not valid, it has shape conflict
            # under the
            return None
        print('$$$$$$$$$$$')
        # replace the original conv/Linuer layers
        for name, module in model.named_modules():
            if not (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
                continue
            ori_in_channels = module.in_channels if isinstance(
                module, nn.Conv2d) else module.in_features
            ori_out_channels = module.out_channels if isinstance(
                module, nn.Conv2d) else module.out_features

            if ori_in_channels == module.input_shape[1] and\
                    ori_out_channels == module.output_shape[1]:
                # no need to replace this layer, only reset its
                # original forward function
                setattr(module, 'forward', self.ori_forwards[name])
                continue
            _new_module = new_module(module)
            super_module, leaf_module = get_module_by_name(model, name)
            setattr(super_module, name.split('.')[-1], _new_module)
        return model

    def generate_channel_cfg(self):
        channel_cfg = {}
        channel_d_sets = self.channel_depen.dependency_sets
        group_d_sets = self.group_depen.dependency
        for _set in channel_d_sets:
            _max_group = max([group_d_sets[name] for name in _set])
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
                sampled_count = int(
                    (random.uniform(0, 1) * f_num) // _max_group)
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
            cfg = yaml.safe_load(cfg_f)
        already_sampled = 0

        while already_sampled < cfg['sample_count']:
            print(already_sampled)
            channel_cfg = self.generate_channel_cfg()
            print(channel_cfg)
            net = self.generate_model(channel_cfg)
            if net is None:
                # generated model is not legal
                continue
            already_sampled += 1
            latency = measure_latency(net, self.dummy_input, cfg)
            _logger.info('Latency : %f', latency)

    def predict(self, model):
        if self.predictor is None:
            return None

    def load(self, ckpath):
        pass

    def export(self, savepath):
        pass
