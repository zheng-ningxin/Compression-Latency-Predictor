# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time
import numpy as np
import onnxruntime

def onnx_run_cpu(onnx_model, dummy_input, run_times=3):
    """
    test the latency of the target model on CPU.
    Parameters
    ----------
    onnx_model: str
        path of the onnx model
    dummy_input: Tensor
        the dummy input of the model to test the latency
    run_times: int
        the numbers of the times to measure the latency
    """
    sess = onnxruntime.InferenceSession(onnx_model)
    sess.set_providers(['CPUExecutionProvider'])
    inputs = sess.get_inputs()
    _in_shapes = []
    _in_types = []
    _in_names = []
    for _input in inputs:
        _in_shapes.append(_input.shape)
        _in_types.append(_input.type)
        _in_names.append(_input.name)
    # generate the inputs for onnx engine
    data = {}
    for _, name in enumerate(_in_names):
        # TODO currently only support 1 input
        # may support multiple inputs model in
        # the future
        data[name] = dummy_input.detach().cpu().numpy()
    latencies = []
    for rid in range(run_times):
        start_time = time.time()
        _ = sess.run(None, data)
        end_time = time.time()
        latencies.append(end_time - start_time)
    return latencies

def onnx_run_gpu(onnx_model, dummy_input, runtimes=3):
    """
    test the latency of the target model on GPU.
    Parameters
    ----------
    onnx_model: str
        path of the onnx model
    dummy_input: Tensor
        the dummy input of the model to test the latency
    run_times: int
        the numbers of the times to measure the latency
    """
    sess = onnxruntime.InferenceSession(onnx_model)
    sess.set_providers(['CUDAExecutionProvider'])
    inputs = sess.get_inputs()
    _in_shapes = []
    _in_types = []
    _in_names = []
    for _input in inputs:
        _in_shapes.append(_input.shape)
        _in_types.append(_input.type)
        _in_names.append(_input.name)
    # generate the inputs for onnx engine
    data = {}
    for i, name in enumerate(_in_names):
        data[name] = dummy_input.detach().cpu().numpy()
    latencies = []
    for rid in range(runtimes):
        start_time = time.time()
        _ = sess.run(None, data)
        end_time = time.time()
        latencies.append(end_time - start_time)
    return latencies

def onnx_run_arm(onnx_model, dummy_input, run_times):
    
    pass
