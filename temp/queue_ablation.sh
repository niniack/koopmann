#!/bin/bash
# Waits for im_jordan_2k_v3, then runs the loss ablation one model at a time
cd /home/nsa325/work/koopmann
while kill -0 1099059 2>/dev/null; do sleep 60; done
grep -q "training exited with code 0" /home/nsa325/work/koopmann/temp/run_jordan_v3.log || { echo "im_jordan_2k_v3 did not finish cleanly, not starting the ablation"; exit 1; }

COMMON="--dataset=imagenet --start_block=6 --end_block=12 --views=3 --num_epochs=200 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 --jordan=True --ae_dim=2049 --ae_hidden=2048"
for run in "cos --lambda_embedding=1" "kl --lambda_kl=1" "both --lambda_embedding=1 --lambda_kl=1"; do
    set -- $run
    name=im_jordan_2k_v3_$1
    shift
    echo "starting $name"
    HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python scripts/tokae/train.py $COMMON "$@" --name=$name > /home/nsa325/work/koopmann/temp/$name.log 2>&1
    echo "$name exited with code $?"
done
