"""Cluster thread limits and persistent diagnostic progress."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CPU_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def configure_cpu_environment() -> int | None:
    """Set library pools before importing numerical libraries, when requested."""
    value = os.environ.get("SHARPNESS_CPUS")
    if value is None:
        return None
    try:
        cpus = int(value)
    except ValueError as error:
        raise ValueError("SHARPNESS_CPUS must be a positive integer.") from error
    if cpus < 1:
        raise ValueError("SHARPNESS_CPUS must be a positive integer.")
    for name in CPU_THREAD_VARIABLES:
        os.environ[name] = str(cpus)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    return cpus


class DiagnosticProgress:
    """Flush stages and Python-process resource measurements independently of W&B."""

    def __init__(self, directory: Path | None, device: Any):
        self.directory = directory
        self.device = device
        self.started = time.monotonic()

    def __call__(self, event: dict[str, Any]) -> None:
        import torch

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports KiB; macOS reports bytes. This is process peak RSS,
        # not a claim about HTCondor's entire job/cgroup memory usage.
        divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
        record = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            **event,
            "process_peak_rss_mib": round(usage.ru_maxrss / divisor, 3),
            "process_cpu_seconds": round(usage.ru_utime + usage.ru_stime, 3),
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        }
        status = Path("/proc/self/status")
        if status.exists():
            try:
                for line in status.read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        record["process_rss_mib"] = round(int(line.split()[1]) / 1024, 3)
                    elif line.startswith("Threads:"):
                        record["process_thread_count"] = int(line.split()[1])
            except OSError:
                pass
        if self.device.type == "cuda":
            for label, measure in (
                ("cuda_allocated_mib", torch.cuda.memory_allocated),
                ("cuda_reserved_mib", torch.cuda.memory_reserved),
                ("cuda_peak_allocated_mib", torch.cuda.max_memory_allocated),
                ("cuda_peak_reserved_mib", torch.cuda.max_memory_reserved),
            ):
                record[label] = round(measure(self.device) / 1024 ** 2, 3)
        line = json.dumps(record, sort_keys=True)
        print("[diagnostic-progress] " + line, flush=True)
        if self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / "diagnostic-progress.jsonl").open("a") as stream:
                stream.write(line + "\n")
                stream.flush()
        except OSError as error:
            # Keep stdout diagnostics even if a job hits its disk limit.
            print(f"Could not save diagnostic progress: {error}", file=sys.stderr, flush=True)
