# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time
import numpy as np
import onnxruntime

def onnx_run_cpu(onnx_model, dummy_input):
    
    sess = onnxruntime.InferenceSession(onnx_model)
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
        x = np.zeros(_in_shapes[i])
        # x = x.astype(_in_types[i])
        # data[name] = dummy_input
        data[name] = dummy_input.detach().cpu().numpy()
    start_time = time.time()
    result = sess.run(None, data)
    end_time = time.time()
    return end_time - start_time

def onnx_run_gpu():
    pass

def onnx_run_arm():
    pass