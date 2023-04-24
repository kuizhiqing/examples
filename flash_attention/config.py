import numpy as np

shape = [2, 1024, 16, 128]
num = 100
scale=1
is_causal=False

seed = 444
dropout = 0
deterministic=True

np_data = "data.npy"
pd_result = "pd.npy"
pt_result = "pt.npy"
pd_result_32 = "pd32.npy"
pt_result_32 = "pt32.npy"

# for fp16
atol = 0.001
rtol = 0.0001

# for fp32
atol = 0.00001
rtol = 0.00001


def get_data():
    with open(np_data, 'rb') as f:
        data = [np.load(f) for i in range(num)]
        return data

