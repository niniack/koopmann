#!/bin/bash
# ImageNet + CC3M, shared modes, no isometric, 2x4096 encoder/decoder, 20 epochs (~13 min each)
SP=/home/nsa325/work/koopmann/temp
cd /home/nsa325/work/koopmann
HF_HUB_OFFLINE=1 $SP/venv/bin/python scripts/tokae/train.py --dataset=imagenet --start_block=6 --end_block=12 --views=3 --tokens=cls_patch_std \
    --num_epochs=20 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 --lambda_isometric=0 --jordan=True --top_k=256 --ae_dim=2049 \
    --ae_hidden=4096 --ae_layers=2 --lambda_embedding=1 --lambda_kl=0.03 --shared_modes=True --extra_dataset=cc3m --name=im_cc3m_2x4k_e20 > $SP/im_cc3m_2x4k_e20.log 2>&1
echo "training exited with code $?"
