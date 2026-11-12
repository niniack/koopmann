#!/bin/bash
# TEMP: the long run of the KAE with multiplicative control (arms mul and mul_half of cls_train.py), with the settings
# and the seed of the baseline run (baseline.sh). 100 epochs at about 0.39 s per step are about 27 hours.
# NOTE: Its own run name, as `<name>_latest.pt` is one file per run name and `base_latest.pt` is the baseline's.
# If the time limit cuts it short, submit it again: it carries on from the last report.
# NOTE: 40 GB as in the baseline: with 20 GB the job is killed for memory in its first minutes (the 28 loader workers).
# Submit from a login shell: sbatch temp/multiply.sh
#SBATCH -J kae_mul
#SBATCH -p nvidia
#SBATCH -q nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 30
#SBATCH --mem=40G
#SBATCH -t 1-08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /home/nsa325/work/koopmann/logs/%x_%j.out

cd /home/nsa325/work/koopmann/temp
# The login shell sets its own HF_HOME, which sbatch copies and .env does not override
export HF_HOME=/scratch/nsa325/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
uv run python cls_train.py --name=base2 --arms=mul,mul_half --total_steps=250000
