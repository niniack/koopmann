#!/bin/bash
# After the isometric-0.1 run (PID 3411567): the same recipe with CLS only as input (768 per block instead of 2304)
SP=/home/nsa325/work/koopmann/temp
while kill -0 3411567 2>/dev/null; do sleep 30; done
cd /home/nsa325/work/koopmann
HF_HUB_OFFLINE=1 $SP/venv/bin/python scripts/tokae/train.py --dataset=imagenet --start_block=6 --end_block=12 --views=3 --tokens=cls \
    --num_epochs=8 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 --lambda_isometric=0.1 --jordan=True --top_k=256 --ae_dim=2049 \
    --ae_hidden=4096 --ae_layers=2 --lambda_embedding=1 --lambda_kl=0.03 --shared_modes=True --name=im_cls_iso0.1_2x4k > $SP/im_cls_iso0.1_2x4k.log 2>&1
echo "training exited with code $?"
