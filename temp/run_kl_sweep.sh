#!/bin/bash
# KL weight sweep on im_jordan256_2k_std settings with the embedding loss: 8 epochs each, to see which weight leaves the other losses alone
cd /home/nsa325/work/koopmann
for kl in 0 0.01 0.03 0.1; do
    HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python scripts/tokae/train.py --dataset=imagenet --start_block=6 --end_block=12 \
        --views=3 --tokens=cls_patch_std --num_epochs=8 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 \
        --jordan=True --top_k=256 --ae_dim=2049 --ae_hidden=2048 --lambda_embedding=1 --lambda_kl=$kl \
        --name=im_std_kl$kl > /home/nsa325/work/koopmann/temp/im_std_kl$kl.log 2>&1
    echo "kl $kl exited with code $?"
done
