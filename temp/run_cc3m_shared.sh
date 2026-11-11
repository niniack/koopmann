#!/bin/bash
# After im_shared_e40 ends: the same, plus CC3M, with ImageNet passes matched (40 epochs), so ~1.75x the steps
SP=/home/nsa325/work/koopmann/temp
while kill -0 3116168 2>/dev/null; do sleep 30; done
cd /home/nsa325/work/koopmann
HF_HUB_OFFLINE=1 $SP/venv/bin/python scripts/tokae/train.py --dataset=imagenet --start_block=6 --end_block=12 --views=3 --tokens=cls_patch_std \
    --num_epochs=40 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 --jordan=True --top_k=256 --ae_dim=2049 --ae_hidden=2048 \
    --lambda_embedding=1 --lambda_kl=0.03 --shared_modes=True --extra_dataset=cc3m --name=im_cc3m_shared_e40 > $SP/im_cc3m_shared_e40.log 2>&1
echo "training exited with code $?"
