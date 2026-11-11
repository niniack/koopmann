#!/bin/bash
# DINOv3 ViT-B/16 + ImageNet-21k images + weight decay 0.1, 12 epochs (twice the CLIP runs' steps, as those were undertrained).
# NOTE: Builds DINOv3's deduplicated 21k cache first; trains only if that succeeded, otherwise removes the partial cache.
# NOTE: transformers comes from a side folder, so the shared venv stays as it is
SP=/home/nsa325/work/koopmann/temp
CACHE=/scratch/nsa325/datasets/w21/clip_cls_patch_std_dinov3-vitb16_train_blocks_5_11.pt
VAL=/scratch/nsa325/datasets/imagenet/clip_cls_patch_std_dinov3-vitb16_val_blocks_5_11.pt
cd /home/nsa325/work/koopmann
[ -f "$VAL" ] || { echo "DINOv3 val cache missing: $VAL"; exit 1; }
if [ ! -f "$CACHE" ]; then
    HF_HUB_OFFLINE=1 PYTHONPATH=$SP/tf_target $SP/venv/bin/python $SP/night/prep_w21.py dinov3-vitb16 || { echo "prep failed, removing partial cache"; rm -f "$CACHE"; exit 1; }
fi
COMMON="--dataset=imagenet --start_block=6 --end_block=12 --views=3 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 --lambda_isometric=0.1 --ae_dim=2049 --ae_hidden=4096 --ae_layers=2 --lambda_embedding=1 --lambda_kl=0.03"
HF_HUB_OFFLINE=1 PYTHONPATH=$SP/tf_target $SP/venv/bin/python scripts/tokae/train.py $COMMON --encoder=dinov3-vitb16 --jordan=True --top_k=256 --shared_modes=True --tokens=cls_patch_std --extra_dataset=w21 --weight_decay=0.1 --num_epochs=12 --name=im_dino_w21_wd
