#!/bin/bash
# TEMP: checks before the long baseline run (baseline.sh). What faster arithmetic does to real CLIP, then a short
# training run that stops and is carried on, then the same in fp32 for its step time. wandb stays offline.
# Submit from a login shell: sbatch temp/baseline_test.sh
#SBATCH -J kae_test
#SBATCH -p nvidia
#SBATCH -q nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o /home/nsa325/work/koopmann/logs/%x_%j.out

cd /home/nsa325/work/koopmann/temp
# The login shell sets its own HF_HOME, which sbatch copies and .env does not override
export HF_HOME=/scratch/nsa325/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
rm -f /scratch/nsa325/koopmann_data/weights_cache/tokae/basetest*_latest.pt
uv run python precision_check.py
uv run python cls_train.py --name=basetest --total_steps=200 --report_every=100 --smoke
uv run python cls_train.py --name=basetest --total_steps=300 --report_every=100 --smoke
uv run python cls_train.py --name=basetest_fp32 --total_steps=200 --report_every=100 --precision=fp32 --smoke
