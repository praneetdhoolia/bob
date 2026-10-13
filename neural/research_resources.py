"""Serialize large offline research workloads within the system RAM budget."""
from contextlib import contextmanager
from pathlib import Path
import time

from filelock import FileLock, Timeout
import psutil

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def research_slot(label, reserve_bytes=3_000_000_000):
    if not 0 <= reserve_bytes < 50_000_000_000:
        raise MemoryError('Research data exceeds the RAM budget; reduce the case limit or stream a smaller subset.')
    lock = FileLock(ROOT/'.runtime/research.lock')
    try:
        lock.acquire(timeout=0)
    except Timeout:
        print(f'Waiting for the current offline research workload before {label}.', flush=True)
        lock.acquire(timeout=-1, poll_interval=1.)
    try:
        waiting = False
        while psutil.virtual_memory().used > 50_000_000_000-reserve_bytes:
            if not waiting:
                print(f'Waiting for system RAM headroom before {label}.', flush=True)
                waiting = True
            time.sleep(1.)
        yield
    finally:
        lock.release()
