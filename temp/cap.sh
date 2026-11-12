#!/bin/bash
# TEMP: is the attention reader limited by its capacity? Three arms of cls_train.py, each `attend` with one change:
# reconstruction weighted 3 (attend_recon3), 256 modes per image (attend_k512), a reader of 3 layers (attend_deep).
# The first 10 epochs of a 100-epoch run with the settings and the seed of lift.sh, so step 25,000 compares with the
# `lift_attend` snapshot of that step. Three arms in place of two, so about 0.5 s per step, about 3 h 30 min.
# NOTE: Its own run name, as `<name>_latest.pt` is one file per run name.
# To train on, drop `--stop_at` and raise the time limit: it carries on from the last report.
# Submit from a login shell: sbatch temp/cap.sh
#SBATCH -J kae_cap
#SBATCH -p nvidia
#SBATCH -q nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 30
#SBATCH --mem=64G
#SBATCH -t 5:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /home/nsa325/work/koopmann/logs/%x_%j.out

cd /home/nsa325/work/koopmann/temp
# The login shell sets its own HF_HOME, which sbatch copies and .env does not override
export HF_HOME=/scratch/nsa325/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
uv run python cls_train.py --name=cap --arms=attend_recon3,attend_k512,attend_deep --total_steps=250000 --stop_at=25000
