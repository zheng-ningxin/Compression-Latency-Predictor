# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import torch

_logger = logging.getLogger(__name__)


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
        model = getattr(model, name)
    leaf_module = getattr(model, name_list[-1])
    return model, leaf_module


def new_conv2d(conv, new_attributes):
    """
    This function will create a new conv layer
    based on the attributes of the original conv
    layer and the modified_attributes. If the
    attribute not specified in the 
    Parameters
    ----------
    conv : torch.nn.Conv2d
        The original conv layer

    Returns
    -------
    torch.nn.Conv2d
        The new conv2d module
    """
    attribute_dict = {
        'in_channels': conv.in_channels,
        'out_channels': conv.out_channels,
        'kernel_size': conv.kernel_size,
        'stride': conv.stride,
        'padding': conv.padding,
        'dilation': conv.dilation,
        'groups': conv.groups,
        'bias': conv.bias is not None,
        'padding_mode': conv.padding_mode
    }
    attribute_dict.update(new_attributes)
    _logger.debug("replace conv2d with in_channels: %d, out_channels: %d",
                  attribute_dict['in_channels'], attribute_dict['out_channels'])
    new_conv = torch.nn.Conv2d(**attribute_dict)

    new_conv.to(conv.weight.device)

    return new_conv

def get_tensors_from(args):
    """
    find all the tensors in the args. args may be a list or a dict
    object.

    Parameters
    ----------
    args: dict or list
        A list or a dict object that may contains Tensors.
    Returns
    -------
    tensors: list
        List of the torch.Tensors find in args
    """
    tensors = []
    if isinstance(args, dict):
        # some layers may return their output as a dict
        # ex. the IntermediateLayerGetter in the face detection jobs.
        for _, val in args.items():
            if isinstance(val, torch.Tensor):
                tensors.append(val)
            else:
                tensors.extend(get_tensors_from(val))
    elif isinstance(args, list) or isinstance(args, tuple):
        # list or tuple
        for item in args:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
            else:
                tensors.extend(get_tensors_from(item))
    elif isinstance(args, torch.Tensor) or isinstance(args, torch.autograd.Variable):
        # if the output is a already a tensor/variable, then return itself
        tensors.append(args)
    return tensors

def measure_latency(model, dummy_input, cfg):
    pass