#!/bin/bash
# Deploy ECBIT Round 1 experiments
SERVER="zhangzhuyu@192.168.10.47"
SSH_CMD="sshpass -p 'zzy4771430' ssh -o StrictHostKeyChecking=no"
for GPU in 0 1 2 3 4; do
  ${SSH_CMD} ${SERVER} "cd ~/ecbit && mkdir -p experiments/logs && nohup /opt/miniconda3/envs/darts/bin/python scripts/worker.py $GPU experiments/configs/round1 > experiments/logs/worker${GPU}.log 2>&1 &"
  echo "GPU $GPU: launched"
done
