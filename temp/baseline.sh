#!/bin/bash
# TEMP: the long baseline run (see cls_train.py): no control, additive control and the MLP, all with mixup.
# Submit from a login shell: sbatch temp/baseline.sh. Submitting it again carries on from the last report.
#SBATCH -J kae_base
#SBATCH -p nvidia
#SBATCH -q nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /home/nsa325/work/koopmann/logs/%x_%j.out

cd /home/nsa325/work/koopmann/temp
# The login shell sets its own HF_HOME, which sbatch copies and .env does not override
export HF_HOME=/scratch/nsa325/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
uv run python cls_train.py --name=base --total_steps=250000
