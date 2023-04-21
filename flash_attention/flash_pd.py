from config import *
import os

if deterministic:
    os.environ["FLAGS_cudnn_deterministic"]="True"
    os.environ["GLOG_vmodule"]="flash_attn_kernel=10,flash_attn_grad_kernel=10,"

import numpy as np
import paddle
from paddle.nn.functional.flash_attention import flash_attention

paddle.seed(seed)

ofile = open(pd_result, 'wb')

def run(query):

    q_pd = paddle.to_tensor(query, dtype="float16")
    q_pd.stop_gradient = False
    
    pd_ret, _ = flash_attention(q_pd, q_pd, q_pd, dropout, is_causal, False)
    pd_ret.backward()
    ret = pd_ret.numpy()

    np.save(ofile, ret)
    np.save(ofile, q_pd.grad)

for i in get_data():
    run(i)

ofile.close()
