# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import yaml
import torch
import random
import torch.nn as nn
import argparse
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.mobilenet import InvertedResidual
from torchvision.models.googlenet import BasicConv2d, Inception, InceptionAux
from torchvision.models.inception import InceptionA, InceptionB, InceptionC, InceptionD, InceptionE

BLOCKS = [BasicBlock, Bottleneck, InvertedResidual, BasicConv2d, Inception,
          InceptionA, InceptionB, InceptionC, InceptionD, InceptionE]

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class FlattenModule(nn.Module):
    def forward(self, x):
        # return x.mean([2, 3])
        return torch.flatten(x, 1)


def get_channel_count(module_list, dummy_input):
    x = dummy_input
    for i, block in enumerate(module_list):
        x = module_list[i](x)
    channel = x.size(1)
    return channel

def append_basicblock(module_list, dummy_input, config):
    in_planes = get_channel_count(module_list, dummy_input)
    planes = random.choice(config['out_channel'])
    stride = random.choice(config['stride'])
    downsample = None

    if stride != 1 or in_planes != planes * BasicBlock.expansion:
        downsample = nn.Sequential(
            conv1x1(in_planes, planes * BasicBlock.expansion, stride),
            nn.BatchNorm2d(planes * BasicBlock.expansion),
        )
    block = BasicBlock(in_planes, planes, stride=stride, downsample=downsample)
    module_list.append(block)

def append_bottleneck(module_list, dummy_input, config):
    in_planes = get_channel_count(module_list, dummy_input)
    planes = random.choice(config['out_channel'])
    stride = random.choice(config['stride'])
    downsample = None

    if stride != 1 or in_planes != planes * Bottleneck.expansion:
        downsample = nn.Sequential(
            conv1x1(in_planes, planes * Bottleneck.expansion, stride),
            nn.BatchNorm2d(planes * Bottleneck.expansion),
        )
    block = Bottleneck(in_planes, planes, stride=stride, downsample=downsample)
    module_list.append(block)

def append_invertedresidual(module_list, dummy_input, config):
    in_planes = get_channel_count(module_list, dummy_input)
    planes = random.choice(config['out_channel'])
    stride = random.choice(config['stride'])
    expand_ratio = random.choice(config['expand_ratio'])
    block = InvertedResidual(in_planes, planes, stride, expand_ratio)
    module_list.append(block)

def append_basicconv2d(module_list, dummy_input, config):
    in_planes = get_channel_count(module_list, dummy_input)
    planes = random.choice(config['out_channel'])
    stride = random.choice(config['stride'])
    block = BasicConv2d(in_planes, planes, stride=stride, kernel_size=(3, 3))
    module_list.append(block)

def append_inception(module_list, dummy_input, config):
    in_channel = get_channel_count(module_list, dummy_input)
    branch_channels = []
    for i in range(6):
        branch_channels.append(random.choice(config['out_channel']))
    args = [in_channel] + branch_channels + [BasicConv2d]
    block = Inception(*args)
    module_list.append(block)

def append_inception_a(module_list, dummy_input, config):
    in_channel = get_channel_count(module_list, dummy_input)
    pool_feature = random.choice(config['out_channel'])
    block = InceptionA(in_channel, pool_feature, BasicConv2d)
    module_list.append(block)

def append_inception_b(module_list, dummy_input, config):
    in_channel = get_channel_count(module_list, dummy_input)
    block = InceptionB(in_channel, BasicConv2d)
    module_list.append(block)

def append_inception_c(module_list, dummy_input, config):
    in_channel = get_channel_count(module_list, dummy_input)
    channel7x7 = random.choice(config['out_channel'])
    block = InceptionC(in_channel, channel7x7, BasicConv2d)
    module_list.append(block)

def append_inception_d(module_list, dummy_input, config):
    in_channel = get_channel_count(module_list, dummy_input)
    block = InceptionD(in_channel, BasicConv2d)
    module_list.append(block)

def append_inception_e(module_list, dummy_input, config):
    in_channel = get_channel_count(module_list, dummy_input)
    block = InceptionE(in_channel, BasicConv2d)
    module_list.append(block)



APPEND_FUNC = [append_basicblock, append_bottleneck, append_invertedresidual, append_basicconv2d,
               append_inception, append_inception_a, append_inception_b, append_inception_c,
               append_inception_d, append_inception_e]

def generate_model(config):
    try:
        n_classes = 1000
        # module_list = nn.ModuleList()
        module_list = []
        dummy_input = torch.ones(1, 3, 224, 224)
        block_count = config['block_count']
        for block_id in range(block_count):
            block_type =  random.randint(0, 9)
            APPEND_FUNC[block_type](module_list, dummy_input, config)
        module_list.append(FlattenModule())
        x = dummy_input
        for block in module_list:
            x = block(x)
        # print(x.size())
        module_list.append(nn.Linear(x.size(1), n_classes))
        model = nn.Sequential(*module_list)
    except Exception as err:
        print(err)
        model = None
    return model

