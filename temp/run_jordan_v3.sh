#!/bin/bash
# Extracts 3 random views of ImageNet once, then trains
cd /home/nsa325/work/koopmann
echo "extracting"
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python /home/nsa325/work/koopmann/temp/extract_views.py || { echo "extraction failed"; exit 1; }
echo "training"
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python scripts/tokae/train.py --dataset=imagenet --start_block=6 --end_block=12     --views=3 --num_epochs=200 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0     --jordan=True --ae_dim=2049 --ae_hidden=2048 --name=im_jordan_2k_v3 > /home/nsa325/work/koopmann/temp/im_jordan_2k_v3.log 2>&1
echo "training exited with code $?"
