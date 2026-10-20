"""Keep the GPU usable for the person at the keyboard while a long job runs.

    from vision.throttle import Throttle
    t = Throttle(share=0.3)      # use about 30% of the GPU
    for batch in ...:
        work(batch)
        t.step()                  # sleeps so that work / (work + sleep) ~= share

Also drops the process to below-normal CPU priority. share=1 disables the pause.
"""
import os
import sys
import time


class Throttle:
    def __init__(self, share=1.0):
        self.share = max(0.05, min(1.0, float(share)))
        self.last = time.perf_counter()
        self.lower_priority()

    @staticmethod
    def lower_priority():
        try:
            if sys.platform == "win32":
                import ctypes
                BELOW_NORMAL = 0x00004000
                ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
            else:
                os.nice(10)
        except Exception:
            pass

    def step(self):
        if self.share >= 1.0:
            return
        now = time.perf_counter()
        busy = now - self.last
        time.sleep(busy * (1.0 - self.share) / self.share)
        self.last = time.perf_counter()
