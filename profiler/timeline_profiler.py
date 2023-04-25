import time
from collections import namedtuple

RECORD = namedtuple('RECORD', ['name', 'stamp', 'duration'])

class TimelineProfiler:
    def __init__(self, verbose=True):
        self.records = []
        self.cursor = 0
        self.verbose = verbose


    def record(self, name):
        ct = time.time()
        duration = ct - self.cursor
        r = RECORD(name, ct, duration)
        self.records.append(r)
        self.cursor = ct
        if self.verbose:
            print(r)

    def summary(self):
        print("== TIMELINE SUMARY BEGIN ==")
        for r in self.records:
            print(r)
        print("==  TIMELINE SUMARY END  ==")

_GLOBAL_TPS = {}
global_tp = TimelineProfiler()

def push_record(name, key=None):
    global _GLOBAL_TPS
    if key and key not in _GLOBAL_TPS:
        _GLOBAL_TPS[key] = TimelineProfiler()

    if key:
        _GLOBAL_TPS[key].record(name)
    else:
        global_tp.record(name)



