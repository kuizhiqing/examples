from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from config import *

import numpy as np

import torch

torch.manual_seed(seed)

ofile = open(pt_result, 'wb')

def run(query):
    q = torch.from_numpy(query).to(device="cuda:0", dtype=torch.float16)
    q.requires_grad = True
    batch_size, seqlen, num_head, head_dim = q.shape
    qt = q.reshape((batch_size * seqlen, q.shape[2], q.shape[3]))
    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q.device)
    torch_ret = flash_attn_unpadded_func(qt, qt, qt, cu_seqlens, cu_seqlens, seqlen, seqlen, dropout, causal=is_causal, deterministic=True)
    torch_ret = torch_ret.reshape((batch_size, seqlen, num_head, head_dim))
    torch_ret.sum().backward()

    torch_ret0 = torch_ret.cpu().detach().numpy()
    torch_ret1 = q.grad.cpu().detach().numpy()

    np.save(ofile, torch_ret0)
    np.save(ofile, torch_ret1)
    
for i in get_data():
    run(i)

ofile.close()
