import time
import random

from timeline_profiler import global_tp

for i in range(3):
    global_tp.record(f"r-{i}")
    time.sleep(random.random()*3)

global_tp.summary()
