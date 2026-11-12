#!/bin/bash
# TEMP: the attention reader with a linear decoder, with and without the real-start rule (arms linear and linear_real
# of cls_train.py, see lift.py). The first 10 epochs of a 100-epoch run with the settings and the seed of lift.sh, so
# step 25,000 compares with the `lift_attend` snapshot of that step. About 0.39 s per step, so about 2 h 45 min.
# NOTE: Its own run name, as `<name>_latest.pt` is one file per run name.
# To train on, drop `--stop_at` and raise the time limit: it carries on from the last report.
# Submit from a login shell: sbatch temp/dec.sh
#SBATCH -J kae_dec
#SBATCH -p nvidia
#SBATCH -q nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 30
#SBATCH --mem=56G
#SBATCH -t 3:45:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /home/nsa325/work/koopmann/logs/%x_%j.out

cd /home/nsa325/work/koopmann/temp
# The login shell sets its own HF_HOME, which sbatch copies and .env does not override
export HF_HOME=/scratch/nsa325/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
uv run python cls_train.py --name=dec --arms=linear,linear_real --total_steps=250000 --stop_at=25000
