#!/bin/bash
# Two hidden layers of 4096 vs one of 2048. Both: shared modes, no isometric loss, 8 epochs, ImageNet only
SP=/home/nsa325/work/koopmann/temp
cd /home/nsa325/work/koopmann
common="--dataset=imagenet --start_block=6 --end_block=12 --views=3 --tokens=cls_patch_std --num_epochs=8 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 --lambda_isometric=0 --jordan=True --top_k=256 --ae_dim=2049 --lambda_embedding=1 --lambda_kl=0.03 --shared_modes=True"
for spec in "im_shared_noiso_2x4k --ae_hidden=4096 --ae_layers=2" "im_shared_noiso --ae_hidden=2048"; do
    set -- $spec
    name=$1; shift
    HF_HUB_OFFLINE=1 $SP/venv/bin/python scripts/tokae/train.py $common "$@" --name=$name > $SP/$name.log 2>&1
    echo "$name exited with code $?"
done
echo "all done"
