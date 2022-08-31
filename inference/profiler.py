import time
from collections import Counter


class Profiler:
    __call_count = Counter()
    __time_elapsed = Counter()
    warmup = 0

    def __init__(self, name, aggregate=False):
        self.name = name
        if not aggregate and Profiler.warmup == 0:
            Profiler.__call_count[self.name] += 1

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.perf_counter()
        self.duration = self.end - self.start
        if Profiler.warmup == 0:
            Profiler.__time_elapsed[self.name] += self.duration
        else:
            Profiler.warmup -= 1

    @classmethod
    def set_warmup(cls, warmup):
        cls.warmup = warmup

    @classmethod
    def reset(cls):
        cls.__call_count.clear()
        cls.__time_elapsed.clear()

    @classmethod
    def get_avg_millis(cls, name):
        call_count = cls.__call_count[name]
        if call_count == 0:
            return 0.
        return cls.__time_elapsed[name] * 1000 / call_count
