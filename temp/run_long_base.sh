#!/bin/bash
# Long run on ImageNet only, with the chosen settings: CLS + patch mean and std, TopK 256, embedding + KL 0.03
cd /home/nsa325/work/koopmann
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python scripts/tokae/train.py --dataset=imagenet --start_block=6 --end_block=12 \
    --views=3 --tokens=cls_patch_std --num_epochs=40 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 \
    --jordan=True --top_k=256 --ae_dim=2049 --ae_hidden=2048 --lambda_embedding=1 --lambda_kl=0.03 \
    --name=im_std_kl0.03_e40 > /home/nsa325/work/koopmann/temp/im_std_kl0.03_e40.log 2>&1
echo "training exited with code $?"
