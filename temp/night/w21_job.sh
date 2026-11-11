#!/bin/bash
# Waits for the ImageNet-21k download, builds the deduplicated extra cache, then trains the control with it mixed in.
# NOTE: Trains only if the cache was built and deduplicated; otherwise removes the partial cache so nothing uses it.
SP=/home/nsa325/work/koopmann/temp
CACHE=/scratch/nsa325/datasets/w21/clip_cls_patch_std_ViT-B-32_train_blocks_5_11.pt
while kill -0 186047 2>/dev/null; do sleep 30; done
n=$(ls /scratch/nsa325/datasets/imagenet-w21-wds/imagenet_w21-train-*.tar | wc -l)
[ "$n" -eq 200 ] || { echo "only $n of 200 shards downloaded"; exit 1; }
cd /home/nsa325/work/koopmann
HF_HUB_OFFLINE=1 $SP/venv/bin/python $SP/night/prep_w21.py || { echo "prep failed, removing partial cache"; rm -f "$CACHE"; exit 1; }
COMMON="--dataset=imagenet --start_block=6 --end_block=12 --views=3 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 --lambda_isometric=0.1 --ae_dim=2049 --ae_hidden=4096 --ae_layers=2 --lambda_embedding=1 --lambda_kl=0.03"
# NOTE: 6 epochs over ImageNet plus ~1.16M extra images is about the control's number of steps (8 epochs over ImageNet)
HF_HUB_OFFLINE=1 $SP/venv/bin/python scripts/tokae/train.py $COMMON --jordan=True --top_k=256 --shared_modes=True --tokens=cls_patch_std --extra_dataset=w21 --num_epochs=6 --weight_decay=0.1 --name=im_w21_wd_2x4k
