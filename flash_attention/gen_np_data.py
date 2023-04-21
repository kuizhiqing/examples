from config import *

import numpy as np

if deterministic:
    np.random.seed(seed)

with open(np_data, 'wb') as f:
    for i in range(num):
        q = scale*np.random.random(shape)
        np.save(f, q)

print(f"Generate random data to {np_data}")
