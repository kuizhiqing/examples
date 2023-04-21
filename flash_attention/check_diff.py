import numpy as np
from config import *

pd_result = "pd32.npy"
pt_result = "pd.npy"

pdf = open(pd_result, 'rb')
ptf = open(pt_result, 'rb')

for i in range(num * 2):
    print(int(i/2), "forward " if i % 2 == 0 else "backward", "--"*10)
    ret_pd = np.load(pdf)
    ret_pt = np.load(ptf)
    try:
       np.testing.assert_allclose(ret_pd, ret_pt, atol=atol, rtol=rtol)
    except Exception as e:
       print(e)
       continue

pdf.close()
ptf.close()
