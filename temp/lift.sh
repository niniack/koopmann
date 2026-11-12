#!/bin/bash
# TEMP: the long run of the KAEs whose encoder reads the patch tokens with the CLS token (arms attend and mixer of
# cls_train.py, see lift.py), with the settings and the seed of the baseline run (baseline.sh). 100 epochs at about
# 0.39 s per step are about 27 hours.
# NOTE: Its own run name, as `<name>_latest.pt` is one file per run name.
# If the time limit cuts it short, submit it again: it carries on from the last report.
# NOTE: 56 GB. With 40 GB (the baseline's) the job was killed for memory after 62,500 steps, while it saved its resume point.
# Submit from a login shell: sbatch temp/lift.sh
#SBATCH -J kae_lift
#SBATCH -p nvidia
#SBATCH -q nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 30
#SBATCH --mem=56G
#SBATCH -t 1-08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /home/nsa325/work/koopmann/logs/%x_%j.out

cd /home/nsa325/work/koopmann/temp
# The login shell sets its own HF_HOME, which sbatch copies and .env does not override
export HF_HOME=/scratch/nsa325/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
uv run python cls_train.py --name=lift --arms=attend,mixer --total_steps=250000
