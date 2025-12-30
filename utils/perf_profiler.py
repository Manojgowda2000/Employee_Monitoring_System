import time
import psutil
from collections import defaultdict

class PerfProfiler:
    def __init__(self):
        self.proc = psutil.Process()
        self.records = defaultdict(list)

    def start(self, name):
        return {
            "name": name,
            "t0": time.perf_counter(),
            "cpu0": self.proc.cpu_percent(interval=None)
        }

    def end(self, ctx):
        t1 = time.perf_counter()
        cpu1 = self.proc.cpu_percent(interval=None)

        elapsed_ms = (t1 - ctx["t0"]) * 1000
        cpu_used = max(cpu1 - ctx["cpu0"], 0.0)

        self.records[ctx["name"]].append({
            "time_ms": elapsed_ms,
            "cpu_percent": cpu_used
        })

    def summary(self):
        out = {}
        for k, vals in self.records.items():
            avg_time = sum(v["time_ms"] for v in vals) / len(vals)
            avg_cpu = sum(v["cpu_percent"] for v in vals) / len(vals)
            out[k] = {
                "avg_time_ms": round(avg_time, 2),
                "avg_cpu_percent": round(avg_cpu, 2),
                "frames": len(vals)
            }
        return out