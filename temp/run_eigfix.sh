#!/bin/bash
# One change each against nek1pwi8 (im_std_kl0.03, 8 epochs): shared modes, per-block centering, per-block mean and std
SP=/home/nsa325/work/koopmann/temp
cd /home/nsa325/work/koopmann
common="--dataset=imagenet --start_block=6 --end_block=12 --views=3 --tokens=cls_patch_std --num_epochs=8 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 --jordan=True --top_k=256 --ae_dim=2049 --ae_hidden=2048 --lambda_embedding=1 --lambda_kl=0.03"
for spec in "im_std_shared --shared_modes=True" "im_std_center --standardization=center" "im_std_block --standardization=block"; do
    set -- $spec
    name=$1; shift
    HF_HUB_OFFLINE=1 $SP/venv/bin/python scripts/tokae/train.py $common "$@" --name=$name > $SP/$name.log 2>&1
    echo "$name exited with code $?"
done
echo "all done"
