from config import *
import os

if deterministic:
    os.environ["FLAGS_cudnn_deterministic"]="True"
    os.environ["GLOG_vmodule"]="flash_attn_kernel=10,flash_attn_grad_kernel=10"

import numpy as np
import paddle

paddle.seed(seed)

def attention_naive(q, k, v, causal=False):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    p = (
        paddle.incubate.softmax_mask_fuse_upper_triangle(s)
        if causal
        else paddle.nn.functional.softmax(s)
    )
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])

ofile = open(pd_result, 'wb')

def run(query):

    q_pd = paddle.to_tensor(query, dtype="float32")
    q_pd.stop_gradient = False
    
    pd_ret = attention_naive(q_pd, q_pd, q_pd, is_causal)
    pd_ret.backward()
    ret = pd_ret.numpy()

    np.save(ofile, ret)
    np.save(ofile, q_pd.grad)

for i in get_data():
    run(i)

ofile.close()
