import numpy as np
from config import *

pd_result = "pd.npy"
pt_result = "pt32.npy"

pdf = open(pd_result, 'rb')
ptf = open(pt_result, 'rb')

def diff_summary(a, b):
    ndiff = np.count_nonzero(np.abs(a) * rtol + atol < np.abs(a - b))
    npos = np.count_nonzero(a > b)
    nneg = np.count_nonzero(a < b)
    madiff = np.max(np.abs(a - b))
    mrdiff = np.max(np.abs(a - b) / np.maximum(a, b))
    #print(f"{ndiff} {a.size} {ndiff/a.size} {madiff} {mrdiff}")
    print(f"{ndiff/a.size} {madiff} {mrdiff}")
    #print(f"{npos} {nneg}")

for i in range(num * 2):
    #print(int(i/2), "forward " if i % 2 == 0 else "backward", "--"*10)
    ret_pd = np.load(pdf)
    ret_pt = np.load(ptf)
    if i % 2 == 0:
        continue
    diff_summary(ret_pd, ret_pt)
    '''
    try:
       np.testing.assert_allclose(ret_pd, ret_pt, atol=atol, rtol=rtol)
    except Exception as e:
       print(e)
       continue
    '''

pdf.close()
ptf.close()
