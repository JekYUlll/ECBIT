#!/usr/bin/env python3
"""ECBIT worker script for multi-GPU experiment dispatch.

Routes stateless/PyPOTS models to evaluate_impute.py and neural models to
train_impute.py. Per-run lock files make duplicate worker launches harmless.
"""
import os
import argparse
import subprocess
import yaml
from pathlib import Path

ROOT = Path.home() / "ecbit"
METRICS = ROOT / "experiments/results/metrics"
LOGS = ROOT / "experiments/logs"

STATELESS_MODELS = {"linear_interp", "locf", "era5_direct", "saits", "brits"}
NEURAL_MODELS = {"itransformer", "ecbit"}
MODEL_PRIORITY = {
    "linear_interp": 0,
    "locf": 1,
    "era5_direct": 2,
    "itransformer": 3,
    "ecbit": 4,
    "saits": 8,
    "brits": 9,
}

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_runner(config):
    """Determine runner from config. Uses 'runner' field if present, else infers from model name."""
    runner = config.get("runner", "")
    if runner:
        return runner
    model_name = str(config.get("model", {}).get("name", ""))
    if model_name in STATELESS_MODELS:
        return "evaluate"
    if model_name in NEURAL_MODELS:
        return "train"
    return "unknown"

def get_run_name(config_path):
    return Path(config_path).stem

def config_sort_key(path):
    config = load_config(path)
    model_name = str(config.get("model", {}).get("name", ""))
    return (MODEL_PRIORITY.get(model_name, 99), Path(path).name)

def result_path(config, name):
    out_dir = config.get("output_dir")
    if out_dir:
        return ROOT / out_dir / "result.json"
    return METRICS / name / "result.json"

def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

def acquire_lock(lock_path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            pid = int(lock_path.read_text().strip())
        except (OSError, ValueError):
            pid = -1
        if pid > 0 and process_alive(pid):
            return False
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker_id", type=int)
    parser.add_argument("round_dir", nargs="?", default="experiments/configs/round1")
    parser.add_argument("--world-size", type=int, default=5)
    parser.add_argument("--cuda-id", type=int, default=None)
    args = parser.parse_args()
    worker_id = args.worker_id
    cuda_id = args.cuda_id if args.cuda_id is not None else worker_id
    round_dir = args.round_dir
    configs = sorted((ROOT / round_dir).glob("*.yaml"), key=config_sort_key)

    # Skip configs that already have result.json
    remaining = []
    for cp in configs:
        name = Path(cp).stem
        config = load_config(cp)
        if result_path(config, name).exists():
            continue
        remaining.append(str(cp))

    my_configs = remaining[worker_id::args.world_size]

    print(
        f"worker {worker_id}/{args.world_size} on cuda {cuda_id}: {len(my_configs)} configs",
        flush=True,
    )
    if not my_configs:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)

    ok = fail = skipped = 0
    for i, cp in enumerate(my_configs):
        name = Path(cp).stem
        config = load_config(cp)
        runner = get_runner(config)
        model_name = config.get("model", {}).get("name", "")

        out_json = result_path(config, name)
        lock_path = out_json.parent / ".lock"
        if out_json.exists():
            skipped += 1
            continue
        if not acquire_lock(lock_path):
            print(f"[{i+1}/{len(my_configs)}] SKIP locked {name}", flush=True)
            skipped += 1
            continue

        print(f"[{i+1}/{len(my_configs)}] {name} ({model_name}, {runner})", flush=True)

        try:
            if runner == "evaluate":
                out_json.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    "python", "src/evaluate_impute.py",
                    "--config", cp,
                    "--split", "test",
                    "--output", str(out_json),
                ]
            else:
                cmd = [
                    "python", "src/train_impute.py",
                    "--config", cp,
                ]

            with open(LOGS / f"{name}.log", "w") as log_f:
                rc = subprocess.run(
                    cmd,
                    cwd=str(ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT
                ).returncode
        except Exception as exc:
            rc = 1
            with open(LOGS / f"{name}.log", "a") as log_f:
                print(f"WORKER_EXCEPTION: {type(exc).__name__}: {exc}", file=log_f)
        finally:
            if not out_json.exists():
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

        if rc == 0 and out_json.exists():
            ok += 1
        else:
            fail += 1
        print(f"[{i+1}/{len(my_configs)}] {name} rc={rc} ok={ok} fail={fail} skipped={skipped}", flush=True)

    print(
        f"worker {worker_id}/{args.world_size} on cuda {cuda_id} DONE: "
        f"ok={ok} fail={fail} skipped={skipped}",
        flush=True,
    )

if __name__ == "__main__":
    main()
