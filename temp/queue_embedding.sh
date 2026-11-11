#!/bin/bash
# Waits for the three Jordan runs, then trains one model at a time
cd /home/nsa325/work/koopmann
while kill -0 968387 2>/dev/null || kill -0 968388 2>/dev/null || kill -0 968389 2>/dev/null; do sleep 30; done
COMMON="--dataset=imagenet --start_block=6 --end_block=12 --views=1 --num_epochs=100 --lambda_variance=0 --lambda_covariance=0 --ae_dim=2048 --ae_hidden=2048"
echo "starting im_2k_emb1"
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python scripts/tokae/train.py $COMMON --lambda_embedding=1 --name=im_2k_emb1 > /home/nsa325/work/koopmann/temp/im_2k_emb1.log 2>&1
echo "im_2k_emb1 exited with code $?"
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python scripts/tokae/train.py $COMMON --lambda_embedding=10 --name=im_2k_emb10 > /home/nsa325/work/koopmann/temp/im_2k_emb10.log 2>&1
echo "im_2k_emb10 exited with code $?"
