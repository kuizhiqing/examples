import time
import sys
from collections import namedtuple

RECORD = namedtuple('RECORD', ['name', 'stamp', 'offset', 'duration'])

class TimelineProfiler:
    """
    Usage
    from timeline_profiler import push_record

    push_record("start")
    """

    def __init__(self, verbose=True, self_record=True):
        self.records = []
        self.cursor = 0
        self.pos = time.time()
        self.verbose = verbose
        if self_record:
            self.record("init")

    def record(self, name):
        ct = time.time()
        duration = ct - self.cursor
        offset = ct - self.pos
        r = RECORD(name, ct, offset, duration)
        self.records.append(r)
        self.cursor = ct
        if self.verbose:
            sys.stderr.write(str(r))
            sys.stderr.write("\n")

    def summary(self):
        sys.stderr.write("== TIMELINE SUMARY BEGIN ==\n")
        for r in self.records:
            sys.stderr.write(str(r))
            sys.stderr.write("\n")
        sys.stderr.write("== TIMELINE SUMARY END ==\n")

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



