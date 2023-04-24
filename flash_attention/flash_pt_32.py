from config import *
import torch.nn as nn

import numpy as np

import torch

torch.manual_seed(seed)

def upper_triangle_mask(size):
    """
    Returns an upper triangular matrix of -inf, with zeros on the diagonal and ones above it.
    """
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)

def attention_naive(q, k, v, causal=False):
    qt = torch.transpose(q, 1, 2)
    kt = torch.transpose(k, 1, 2)
    vt = torch.transpose(v, 1, 2)

    #mask = upper_triangle_mask(qt.shape[1]).to(q.device)

    s = torch.matmul(qt, kt.transpose(2,3)) / (q.shape[-1] ** 0.5)
    attn = nn.functional.softmax(s, dim=3)
    st = torch.matmul(attn, vt)
    return torch.transpose(st, 1, 2)

ofile = open(pt_result_32, 'wb')

def run(query):
    q = torch.from_numpy(query).to(device="cuda:0", dtype=torch.float32)
    q.requires_grad = True

    torch_ret = attention_naive(q, q, q)
    torch_ret.sum().backward()

    torch_ret0 = torch_ret.cpu().detach().numpy()
    torch_ret1 = q.grad.cpu().detach().numpy()

    np.save(ofile, torch_ret0)
    np.save(ofile, torch_ret1)
    
for i in get_data():
    run(i)

ofile.close()
