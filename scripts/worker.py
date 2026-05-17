#!/usr/bin/env python3
"""ECBIT worker script for multi-GPU experiment dispatch."""
import subprocess, os, sys
from pathlib import Path

ROOT = Path.home() / "ecbit"
METRICS = ROOT / "experiments/results/metrics"
LOGS = ROOT / "experiments/logs"

def main():
    gpu_id = int(sys.argv[1])
    round_dir = sys.argv[2] if len(sys.argv) > 2 else "experiments/configs/round1"
    configs = sorted((ROOT / round_dir).glob("*.yaml"))
    remaining = [str(c) for c in configs
                 if not (METRICS / f"{c.stem}.json").exists()]
    my_configs = remaining[gpu_id::5]
    
    print(f"GPU {gpu_id}: {len(my_configs)} configs")
    if not my_configs:
        return
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ok = fail = 0
    for i, cp in enumerate(my_configs):
        name = Path(cp).stem
        print(f"[{i+1}/{len(my_configs)}] {name}")
        with open(LOGS / f"{name}.log", 'w') as log_f:
            rc = subprocess.run(
                ["python", "src/train_impute.py", "--config", cp],
                cwd=str(ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT).returncode
        if rc == 0: ok += 1
        else: fail += 1
    print(f"GPU {gpu_id} DONE: ok={ok} fail={fail}")

if __name__ == "__main__":
    main()
