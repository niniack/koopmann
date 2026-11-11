#!/bin/bash
# TEMP: a 20-minute check of a GPU node (see hpc_check.py). Submit from a login shell: sbatch temp/hpc_check.sh
#SBATCH -J kae_check
#SBATCH -p nvidia
#SBATCH -q ebrainccs
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 0:20:00
#SBATCH -o /home/nsa325/work/koopmann/logs/%x_%j.out

cd /home/nsa325/work/koopmann/temp
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "internet: $(curl -s -m 10 -o /dev/null -w '%{http_code}' https://api.wandb.ai || echo none)"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python hpc_check.py 16
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python hpc_check.py 30
