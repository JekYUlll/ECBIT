#!/usr/bin/env python3
"""ECBIT worker script for multi-GPU experiment dispatch.

Routes stateless/PyPOTS models to evaluate_impute.py and neural models to train_impute.py.
"""
import subprocess, os, sys, yaml
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

def main():
    gpu_id = int(sys.argv[1])
    round_dir = sys.argv[2] if len(sys.argv) > 2 else "experiments/configs/round1"
    configs = sorted((ROOT / round_dir).glob("*.yaml"), key=config_sort_key)

    # Skip configs that already have result.json
    remaining = []
    for cp in configs:
        name = Path(cp).stem
        config = load_config(cp)
        if result_path(config, name).exists():
            continue
        remaining.append(str(cp))

    my_configs = remaining[gpu_id::5]

    print(f"GPU {gpu_id}: {len(my_configs)} configs")
    if not my_configs:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Reduce DataLoader workers to avoid "Too many open files"
    # 5 GPU workers × 2 num_workers = 10 DataLoader workers total — safe within 1024 fd limit
    NUM_WORKERS_SAFE = 2
    ok = fail = skipped = 0
    for i, cp in enumerate(my_configs):
        name = Path(cp).stem
        config = load_config(cp)
        runner = get_runner(config)
        model_name = config.get("model", {}).get("name", "")

        print(f"[{i+1}/{len(my_configs)}] {name} ({model_name}, {runner})")

        if runner == "evaluate":
            out_json = result_path(config, name)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            # Route to evaluate_impute.py — stateless, no GPU needed for simple baselines
            cmd = [
                "python", "src/evaluate_impute.py",
                "--config", cp,
                "--split", "test",
                "--output", str(out_json),
            ]
        else:
            # Neural model — train on GPU
            out_dir = METRICS / name
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python", "src/train_impute.py",
                "--config", cp,
            ]

        with open(LOGS / f"{name}.log", 'w') as log_f:
            rc = subprocess.run(
                cmd,
                cwd=str(ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT
            ).returncode

        if rc == 0:
            ok += 1
        else:
            fail += 1

    print(f"GPU {gpu_id} DONE: ok={ok} fail={fail}")

if __name__ == "__main__":
    main()
