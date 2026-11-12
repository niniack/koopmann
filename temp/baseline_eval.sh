#!/bin/bash
# TEMP: the fixed test (baseline_eval.py) of a run's saved weights. Arguments are handed to the script.
# Submit from a login shell: sbatch temp/baseline_eval.sh --files="base_*"
#SBATCH -J kae_eval
#SBATCH -p nvidia
#SBATCH -q nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 0:45:00
#SBATCH -o /home/nsa325/work/koopmann/logs/%x_%j.out

cd /home/nsa325/work/koopmann/temp
# The login shell sets its own HF_HOME, which sbatch copies and .env does not override
export HF_HOME=/scratch/nsa325/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
uv run python baseline_eval.py "$@"
