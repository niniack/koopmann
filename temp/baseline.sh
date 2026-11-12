#!/bin/bash
# TEMP: the long baseline run (see cls_train.py): no control and the MLP, both with mixup.
# NOTE: The additive-control arm was stopped at step 35,000 (its resume point is kept in base_latest_with_add.pt).
# 100 epochs at about 0.45 s per step are about 32 hours. If the time limit cuts it short, submit it again: it carries
# on from the last report. Submit from a login shell: sbatch temp/baseline.sh
#SBATCH -J kae_base
#SBATCH -p nvidia
#SBATCH -q nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 30
#SBATCH --mem=40G
#SBATCH -t 1-16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /home/nsa325/work/koopmann/logs/%x_%j.out

cd /home/nsa325/work/koopmann/temp
# The login shell sets its own HF_HOME, which sbatch copies and .env does not override
export HF_HOME=/scratch/nsa325/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
uv run python cls_train.py --name=base --arms=none,mlp --total_steps=250000
