# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import argparse
import nni
import numpy as np
from nni.compression.torch import L1FilterPruner
from nni.compression.torch.speedup import ModelSpeedup
from nni._graph_utils import build_module_graph

Dirs = ['mobiletnet_v2_k80', 'resnet34_k80', 'vgg11_k80', 'vgg19_k80']

total_node_count = 0
total_edge_count = 0
total_graph_count = 0
adj_f = 'Latency_A.txt'
graph_indicator_f = 'Latency_graph_indicator.txt'
graph_label_f = 'Latency_graph_labels.txt'
node_label_f = 'Latency_node_label.txt'
graph_attr_f ='Latency_graph_attributes.txt'
node_attr_f = 'Latency_node_attributes.txt'

Prefix = './data'

def write_graph_indicator(n, graphid):
    with open(graph_indicator_f, 'a') as graph_indicator:
        for i in range(n):
            graph_indicator.write('%d\n'%graphid)

def write_graph_adjacent(nodeid_map, torch_graph):
    with open(adj_f, 'a') as adj:
        for node in torch_graph.nodes_py.nodes_op:
            unique_name = node.unique_name
            successors = torch_graph.find_successors(unique_name)
            for successor in successors:
                u_nid, v_nid = nodeid_map[unique_name], nodeid_map[successor]
                adj.write('%d, %d\n' % (u_nid, v_nid))


def write_node_label(torch_graph):
    with open(node_label_f, 'a') as node_label:
        for op in torch_graph.nodes_py.nodes_op:
            node_label.write('%s\n' % op.op_type)

def write_graph_label(label):
    with open(graph_label_f, 'a') as graph_label:
        graph_label.write('%s\n' % label)

def write_graph_attribute(avg, std):
    with open(graph_attr_f, 'a') as graph_attr:
        graph_attr.write('%.5f, %.5f\n' % (avg, std))

def get_module_by_name(model, module_name):
    """
    Get a module specified by its module name

    Parameters
    ----------
    model : pytorch model
        the pytorch model from which to get its module
    module_name : str
        the name of the required module

    Returns
    -------
    module, module
        the parent module of the required module, the required module
    """
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)
        else:
            return None, None
    if hasattr(model, name_list[-1]):
        leaf_module = getattr(model, name_list[-1])
        return model, leaf_module
    else:
        return None, None


def write_node_attr(torch_graph):
    with open(node_attr_f, 'a') as node_attr:
        op_nodes = torch_graph.nodes_py.nodes_op
        bound_model = torch_graph.bound_model
        
        traced_graph = torch_graph.trace.graph
        debugname_to_value = {}
        for node in traced_graph.nodes():
            for _input in node.inputs():
                debug_name = _input.debugName()
                if debug_name not in debugname_to_value:
                    debugname_to_value[debug_name] = _input
            for _output in node.outputs():
                debug_name = _output.debugName()
                if debug_name not in debugname_to_value:
                    debugname_to_value[debug_name] = _output

        for op in op_nodes:
            in_shapes = []
            for in_t in op.inputs:
                if in_t in torch_graph.output_to_node or in_t in torch_graph.nodes_py.nodes_io:
                    c_node = debugname_to_value[in_t]

                    if isinstance(c_node.type(), torch._C.TensorType):
                        shape = tuple(c_node.type().sizes())
                        in_shapes.append(shape)

            node_attr.write(str(in_shapes))
            if op.type == 'module':
            
                module_name = op.name
                _, module = get_module_by_name(bound_model, module_name)
                other_attr = ''
                if isinstance(module, nn.Conv2d):
                    other_attr = ' {} {} {} {} {}'.format(module.in_channels, module.out_channels, module.groups, module.kernel_size, module.bias is not None)
                elif isinstance(module, nn.Linear):
                    other_attr = ' {} {} {}'.format(module.in_features, module.out_features, module.bias is not None)
                elif isinstance(module, nn.BatchNorm2d):
                    other_attr = ' {}'.format(module.num_features)
                node_attr.write(other_attr)
            node_attr.write('\n')


dummy_input = torch.ones(16, 3, 224, 224).cuda()

def build_model(model_class, channelcfg):
    model = model_class(pretrained=True).cuda()
    # dummy_input = torch.ones(16, 3, 224, 224).cuda()
    cfglist = []
    pos = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            sparsity = 1-channelcfg[pos]/module.out_channels
            if sparsity > 0:
                cfglist.append({'sparsity':sparsity + 1e-5, 'op_names':[name], 'op_types': ['Conv2d']}) 
            pos += 1
    pruner = L1FilterPruner(model, cfglist, dummy_input=dummy_input, dependency_aware=True)
    pruner.compress()
    pruner.export_model('./model.pth', './mask')
    pruner._unwrap_model()
    del pruner
    ms = ModelSpeedup(model, dummy_input, './mask')
    ms.speedup_model()
    del ms
    pos = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(module.out_channels,  channelcfg[pos])
            assert module.out_channels == channelcfg[pos]
            pos += 1
    # del dummy_input
    torch.cuda.empty_cache()
    
    return model



construct_model_func = [models.mobilenet_v2, models.resnet34, models.vgg11, models.vgg19]



for modeltype, dir in enumerate(Dirs):
    dir_path = os.path.join(Prefix, dir)
    files = os.listdir(dir_path)
    print(files)
    for filename in files:
        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'r') as jf:
            data = json.load(jf)
            for model in data:
                total_graph_count += 1
                model_cfg = model[0]
                latencies = model[1]
                bound_model = build_model(construct_model_func[modeltype], model_cfg)
                # print(bound_model)
                torch_graph = build_module_graph(bound_model, dummy_input)
                op_nodes = torch_graph.nodes_py.nodes_op
                n_count = len(op_nodes)
                # write the graph indicator
                write_graph_indicator(n_count, total_graph_count)
                node_id = {}
                for i in range(1, n_count+1):
                    cur_nodeid = total_node_count + i
                    unique_name = op_nodes[i-1].unique_name
                    node_id[unique_name] = cur_nodeid
                # write the graph adjacent matrix
                write_graph_adjacent(node_id, torch_graph)
                write_node_label(torch_graph)
                write_graph_label(str(type(bound_model)))
                model_latency = latencies['model'][2:-2]
                latency_mean, latency_std = np.mean(model_latency), np.std(model_latency)
                write_graph_attribute(latency_mean, latency_std)
                write_node_attr(torch_graph)
                total_node_count += n_count


