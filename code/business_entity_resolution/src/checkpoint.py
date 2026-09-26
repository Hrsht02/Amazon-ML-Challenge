"""Crash-safe checkpoints and resource telemetry for long-running runs."""

from __future__ import annotations
import json, os, pickle, tempfile, time
from pathlib import Path
from typing import Any
import numpy as np

def atomic_json(path: Path, obj: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_pickle(path: Path, obj: Any, compression: bool = True) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if compression:
        import gzip
        with gzip.open(tmp, "wb", compresslevel=1) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def load_pickle(path: Path, compression: bool = True) -> Any:
    if compression:
        import gzip
        with gzip.open(path, "rb") as f: return pickle.load(f)
    with open(path, "rb") as f: return pickle.load(f)

def mark_done(run_dir: Path, stage: str, **meta: Any) -> None:
    atomic_json(Path(run_dir) / f".done_{stage}.json", {
        "stage": stage, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta
    })

def is_done(run_dir: Path, stage: str) -> bool:
    return (Path(run_dir) / f".done_{stage}.json").exists()

def resource_snapshot() -> dict:
    out = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out.update({
            "ram_total_gb": round(vm.total / 2**30, 2),
            "ram_available_gb": round(vm.available / 2**30, 2),
            "ram_used_pct": round(vm.percent, 1),
            "cpu_pct": round(psutil.cpu_percent(interval=None), 1),
        })
    except Exception: pass
    try:
        import torch
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["gpu_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            out["gpu_free_gb"] = round(free / 2**30, 2)
            out["gpu_total_gb"] = round(total / 2**30, 2)
    except Exception: out["cuda_available"] = False
    return out

def print_resources(prefix: str = "[RESOURCES]") -> None:
    snap = resource_snapshot()
    if snap: print(prefix, json.dumps(snap, sort_keys=True), flush=True)

def save_numpy(path: Path, arr: np.ndarray) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, arr)
    actual = Path(str(tmp) + ".npy") if not tmp.name.endswith(".npy") else tmp
    os.replace(actual, path)
