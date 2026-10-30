"""Run the paper-listed Mamba4KT hyperparameter grid on the frozen AS09 fold."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
UNIKT = ROOT / "UniKT"
CONFIG = Path(
    os.environ.get(
        "KT_GRID_CONFIG", str(ROOT / "mamba4kt_fold0_seed42.yaml")
    )
).resolve()
OUTPUT = Path(
    os.environ.get("KT_GRID_OUTPUT", str(ROOT / "grid_unikt"))
).resolve()
PROFILE = os.environ.get("KT_GRID_PROFILE", "unikt_current")
LOGS = OUTPUT / "logs"
RESOURCES = OUTPUT / "resources"

HIDDEN_SIZES = tuple(
    int(value)
    for value in os.environ.get("KT_GRID_HIDDEN_SIZES", "64,128,256").split(",")
)
LEARNING_RATES = (0.003, 0.002, 0.001, 0.0001)


def newest_run(started_ns: int) -> Path:
    candidates = [
        path
        for path in (UNIKT / "runs" / "normal").iterdir()
        if path.is_dir() and path.stat().st_mtime_ns >= started_ns
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one new run directory, found {candidates}")
    return candidates[0]


def read_best(run_dir: Path) -> dict[str, float | int | str]:
    with (run_dir / "metrics_val.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    best_auc = max(rows, key=lambda row: float(row["auc"]))
    best_acc = max(rows, key=lambda row: float(row["acc"]))
    return {
        "epochs_completed": len(rows),
        "best_auc_epoch_zero_based": int(best_auc["epoch"]),
        "best_validation_auc": float(best_auc["auc"]),
        "best_acc_epoch_zero_based": int(best_acc["epoch"]),
        "best_validation_acc": float(best_acc["acc"]),
        "run_dir": str(run_dir.relative_to(UNIKT)),
        "checkpoint": str((run_dir / "best_model.pth").relative_to(UNIKT)),
    }


def wsl_memory_used_mib() -> int | None:
    """Return memory currently used inside this WSL instance."""
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.split()[0])
        return (values["MemTotal"] - values["MemAvailable"]) // 1024
    except (OSError, KeyError, ValueError):
        return None


def gpu_memory_used_mib() -> int | None:
    """Return total GPU memory in use, including unrelated host processes."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return max(int(value.strip()) for value in output.splitlines())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def process_tree_rss_mib(root_pid: int) -> int | None:
    """Sum the resident memory of a process and its descendants."""
    fields: dict[int, tuple[int, int]] = {}
    try:
        for status_path in Path("/proc").glob("[0-9]*/status"):
            pid = int(status_path.parent.name)
            values: dict[str, int] = {}
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    name, raw = line.split(":", 1)
                    if name in {"PPid", "VmRSS"}:
                        values[name] = int(raw.split()[0])
            if "PPid" in values and "VmRSS" in values:
                fields[pid] = (values["PPid"], values["VmRSS"])
    except OSError:
        return None

    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _) in fields.items():
            if pid not in tree and parent_pid in tree:
                tree.add(pid)
                changed = True
    if root_pid not in fields:
        return None
    return sum(fields[pid][1] for pid in tree if pid in fields) // 1024


def run_monitored(command: list[str], log_path: Path, label: str) -> dict:
    """Run one training job while sampling host-RAM and GPU-memory peaks."""
    env = {
        **os.environ,
        "KT_DATALOADER_WORKERS": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "MALLOC_ARENA_MAX": "2",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    started = time.monotonic()
    baseline_memory = wsl_memory_used_mib()
    baseline_gpu = gpu_memory_used_mib()
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=UNIKT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        peak_memory = baseline_memory
        peak_gpu = baseline_gpu
        peak_process_rss = None
        samples = 0
        while process.poll() is None:
            memory = wsl_memory_used_mib()
            gpu = gpu_memory_used_mib()
            rss = process_tree_rss_mib(process.pid)
            if memory is not None:
                peak_memory = max(peak_memory or 0, memory)
            if gpu is not None:
                peak_gpu = max(peak_gpu or 0, gpu)
            if rss is not None:
                peak_process_rss = max(peak_process_rss or 0, rss)
            samples += 1
            time.sleep(2)
        returncode = process.wait()

    resources = {
        "label": label,
        "returncode": returncode,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "baseline_wsl_memory_mib": baseline_memory,
        "peak_wsl_memory_mib": peak_memory,
        "peak_training_process_tree_rss_mib": peak_process_rss,
        "baseline_gpu_memory_mib": baseline_gpu,
        "peak_gpu_memory_mib": peak_gpu,
        "sample_count": samples,
        "gpu_memory_scope": "whole GPU, including unrelated host processes",
    }
    (RESOURCES / f"{label}.json").write_text(json.dumps(resources, indent=2) + "\n")
    return resources


def main() -> None:
    configured_model = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["model"]
    if PROFILE == "paper_faithful":
        expected = {"separate_qa": 1, "prediction_head": "unikt"}
        actual = {key: configured_model.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"paper_faithful requires {expected}, received {actual}"
            )

    OUTPUT.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    RESOURCES.mkdir(exist_ok=True)
    results_path = OUTPUT / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    completed = {
        (int(row["hidden_size"]), float(row["learning_rate"]))
        for row in results
        if row.get("status") == "completed"
    }

    for hidden_size in HIDDEN_SIZES:
        for learning_rate in LEARNING_RATES:
            key = (hidden_size, learning_rate)
            if key in completed:
                continue
            label = f"d{hidden_size}_lr{learning_rate:g}"
            log_path = LOGS / f"{label}.log"
            command = [
                sys.executable,
                "train.py",
                "--general.progress",
                "none",
                "--general.save_last_checkpoint",
                "false",
                "--config",
                str(CONFIG),
                "--model.d_model",
                str(hidden_size),
                "--model.learning_rate",
                str(learning_rate),
            ]
            started_ns = time.time_ns()
            resources = run_monitored(command, log_path, label)
            if resources["returncode"] != 0:
                result = {
                    "profile": PROFILE,
                    "hidden_size": hidden_size,
                    "learning_rate": learning_rate,
                    "status": "failed",
                    "log": str(log_path.relative_to(ROOT)),
                    "resources": str(
                        (RESOURCES / f"{label}.json").relative_to(ROOT)
                    ),
                }
                results.append(result)
                results_path.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(result), flush=True)
                continue
            result = {
                "profile": PROFILE,
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "status": "completed",
                **read_best(newest_run(started_ns)),
                "log": str(log_path.relative_to(ROOT)),
                "resources": str(
                    (RESOURCES / f"{label}.json").relative_to(ROOT)
                ),
            }
            results.append(result)
            results_path.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
