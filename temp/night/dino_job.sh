#!/bin/bash
# DINOv3 ViT-B/16 with the CLIP control's settings: extracts ImageNet's states (3 views + val) on first use, then trains.
# NOTE: transformers comes from a side folder, so the shared venv stays as it is
SP=/home/nsa325/work/koopmann/temp
cd /home/nsa325/work/koopmann
COMMON="--dataset=imagenet --start_block=6 --end_block=12 --views=3 --num_epochs=8 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 --lambda_isometric=0.1 --ae_dim=2049 --ae_hidden=4096 --ae_layers=2 --lambda_embedding=1 --lambda_kl=0.03"
HF_HUB_OFFLINE=1 PYTHONPATH=$SP/tf_target $SP/venv/bin/python scripts/tokae/train.py $COMMON --encoder=dinov3-vitb16 --jordan=True --top_k=256 --shared_modes=True --tokens=cls_patch_std --name=im_dino_2x4k
