#!/bin/bash
# im_jordan256_2k_patch with the learning rates raised about 2.7x
cd /home/nsa325/work/koopmann
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python scripts/tokae/train.py --dataset=imagenet --start_block=6 --end_block=12 \
    --views=3 --tokens=cls_patch --num_epochs=40 --lr=0.02 --adam_lr=0.002 --lambda_variance=0 --lambda_covariance=0 \
    --jordan=True --top_k=256 --ae_dim=2049 --ae_hidden=2048 --lambda_embedding=1 --lambda_kl=1 \
    --name=im_jordan256_2k_patch_lr > /home/nsa325/work/koopmann/temp/im_jordan256_2k_patch_lr.log 2>&1
echo "training exited with code $?"
