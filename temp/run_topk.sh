#!/bin/bash
# im_jordan_2k_v3_both settings, keeping the top 128 of 1025 modes
cd /home/nsa325/work/koopmann
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python scripts/tokae/train.py --dataset=imagenet --start_block=6 --end_block=12     --views=3 --num_epochs=200 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0     --jordan=True --top_k=256 --ae_dim=2049 --ae_hidden=2048 --lambda_embedding=1 --lambda_kl=1     --name=im_jordan256_2k_both > /home/nsa325/work/koopmann/temp/im_jordan256_2k_both.log 2>&1
echo "training exited with code $?"
