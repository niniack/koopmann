#!/bin/bash
# im_jordan256_2k_patch settings, with the patch tokens' std added to the state at both ends
# Usage: run_im_jordan256_2k_std.sh LR ADAM_LR
cd /home/nsa325/work/koopmann
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python /home/nsa325/work/koopmann/temp/extract_std.py > /home/nsa325/work/koopmann/temp/extract_std.log 2>&1 || exit 1
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python scripts/tokae/train.py --dataset=imagenet --start_block=6 --end_block=12 \
    --views=3 --tokens=cls_patch_std --num_epochs=40 --lr=$1 --adam_lr=$2 --lambda_variance=0 --lambda_covariance=0 \
    --jordan=True --top_k=256 --ae_dim=2049 --ae_hidden=2048 --lambda_embedding=1 --lambda_kl=1 \
    --name=im_jordan256_2k_std > /home/nsa325/work/koopmann/temp/im_jordan256_2k_std.log 2>&1
echo "training exited with code $?"
