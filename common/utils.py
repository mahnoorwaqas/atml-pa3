"""Shared helpers: seeding, IO, timing, GPU memory, parameter counting."""
import json, os, random, time
from contextlib import contextmanager
import numpy as np

SEED = 6304


def seed_everything(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, set): return sorted(o)
    return str(o)


def save_json(obj, path, indent=2):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=_json_default) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@contextmanager
def timer(store: dict = None, key: str = "time"):
    """Wall-clock timer that synchronises CUDA before/after; accumulates into store[key]."""
    _sync(); t = time.perf_counter()
    yield
    _sync(); dt = time.perf_counter() - t
    if store is not None:
        store[key] = store.get(key, 0.0) + dt


def peak_mem_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**30
    except ImportError:
        pass
    return 0.0


def reset_peak_mem():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def path_size_bytes(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            tot += os.path.getsize(os.path.join(r, f))
    return tot


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
