# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import re
import traceback
import copy
import json
import yaml
import random
import logging
import torch
import joblib
import numpy as np
import torch.nn as nn
from torch import Tensor

from nni._graph_utils import TorchModuleGraph
from nni.compression.torch.utils.shape_dependency import ChannelDependency, GroupDependency
from nni.compression.torch import Constrained_L1FilterPruner
from nni.compression.torch import ModelSpeedup
from .util import *
import gc
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class CompressionLatencyDataset:
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
        self.training = self.bound_model.training
        self.bound_model.eval()

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
        self.measured_data = []

    def generate_model(self, cfg):
        """
        generate the models according to the channel_cfg.
        The generated model has the same network architecture
        with self.bound_model, but the out_channels of each
        conv layers are configured according to the channel_cfg.

        Parameters
        ----------
        cfg: list
            cfg for the pruner.
        """

        model = copy.deepcopy(self.bound_model)
        pruner = Constrained_L1FilterPruner(model, cfg, self.dummy_input)
        pruner.compress()
        _tmp_ck_path = os.path.join(self.ck_dir, 'tmp.pth')
        _tmp_mask_path = os.path.join(self.ck_dir, 'mask')
        pruner.export_model(_tmp_ck_path, _tmp_mask_path)
        pruner._unwrap_model()
        ms = ModelSpeedup(model, self.dummy_input, _tmp_mask_path)
        ms.speedup_model()

        try:
            model(self.dummy_input)
            print('Success inference')
        except Exception as err:
            _logger.warn('The updated model may have shape conflicts')
            _logger.warn(err)
            traceback.print_exc()
            # the model is not valid, it has shape conflict
            # under the
            return None

        return model

    def generate_random_cfg(self, cfg):
        
        cfglist = []
        channel_d_sets = self.channel_depen.dependency_sets
        # group_d_sets = self.group_depen.dependency
        for _set in channel_d_sets:
            if random.uniform(0, 1) > 0.7:
                # there is 70% probability that we donnot
                # prune these layers
                continue
            else:
                sparsity = 0
                while sparsity <= 0 or sparsity >= 1.0:
                    sparsity = random.uniform(0, 1)
                for layer in _set:
                    cfglist.append({'op_types': ['Conv2d'], 'op_names': [
                                   layer], 'sparsity': sparsity})
        return cfglist

    def generate_specified_cfg(self, cfg):
        """
        Generate the channel sparsity cfg according to the cfg if needed,
        if the specified sparsity config already run out or there is no
        specified sparsity search space, then return None:

        Parameters
        ----------
        cfg: dict
            the dict object read from the config file
        Returns
        -------
        cfg_list: list or None
            if there are specified configs not visited yet, then return one
            sparsity config in the specified sparsity space, else return None.
        """
        def traverse_cfg_space(pattern_list, space, i, tmp_cfg):
            if i == len(space):
                pattern_list.append(copy.deepcopy(tmp_cfg))
                return
            for pattern_name in space[i]:
                for sparsity in np.arange(space[i][pattern_name]['start'], space[i][pattern_name]['end'], space[i][pattern_name]['step']):
                    tmp_cfg[pattern_name] = round(sparsity, 2)
                    traverse_cfg_space(pattern_list, space, i+1, tmp_cfg )

        if 'specified_sample_space' not in cfg:
            return []
        cfg_patterns = []
        CFGs = []
        for space in cfg['specified_sample_space']:
            traverse_cfg_space(cfg_patterns, space, 0, {})
        print('PATTERNS')
        print(cfg_patterns)
        for cfg_pattern in cfg_patterns:
            cfg_list = []
            for name, _ in self.bound_model.named_modules():
                for pattern in cfg_pattern:
                    if re.match(pattern, name):
                        print(pattern, name)
                        cfg_list.append({'op_types': ['Conv2d'], 'op_names': [
                                   name], 'sparsity': cfg_pattern[pattern]})
            CFGs.append(cfg_list)
        print(CFGs)
        print('Len Specified', len(CFGs))
        return CFGs

    def generate_cfg(self, cfg):
        sparsity_cfgs = []
        sparsity_cfgs.extend(self.generate_specified_cfg(cfg))
        for i in range(len(sparsity_cfgs), cfg['sample_count']):
            sparsity_cfgs.append(self.generate_random_cfg(cfg))
        print('total configs', len(sparsity_cfgs))
        return sparsity_cfgs

    def generate_dataset(self, cfgpath):
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
        ck_dir = cfg.get('checkpoint_dir', './log')
        os.makedirs(ck_dir, exist_ok=True)
        self.ck_dir = ck_dir
        sparsity_cfgs = self.generate_cfg(cfg)
        for already_sampled, cfglist in enumerate(sparsity_cfgs):
            _logger.info('Sample the %d-th model', already_sampled+1)
            print(cfglist)
            # _logger.info('Sparsity config', str(cfglist))
            net = self.generate_model(cfglist)
            if net is None:
                # generated model is not legal
                continue
            already_sampled += 1
            module_level = cfg.get('submodule_level', 0)
            latency = measure_latency(net, self.dummy_input, cfg, level=module_level)
            _logger.info('Latency : %f', sum(latency['model'])/len(latency['model']))
            print('Measured Latency', latency)
            self.measured_data.append((get_channel_list(net), latency))
            del net
            gc.collect()
        # save the measured data to the checkpoint dirpath
        with open(os.path.join(ck_dir, 'raw_data.json'), 'w') as jf:
            json.dump(self.measured_data, jf)
