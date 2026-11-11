#!/bin/bash
# After the ImageNet-only baseline and the download: extract CC3M, then train on ImageNet + CC3M for the same number of steps
until grep -q "training exited" /home/nsa325/work/koopmann/temp/run_long_base_outer.log && grep -q "^done" /home/nsa325/work/koopmann/temp/download_cc3m.log; do sleep 60; done
cd /home/nsa325/work/koopmann
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python /home/nsa325/work/koopmann/temp/extract_cc3m.py > /home/nsa325/work/koopmann/temp/extract_cc3m.log 2>&1 || { echo "extraction failed"; exit 1; }
extra=$(grep "^done" /home/nsa325/work/koopmann/temp/extract_cc3m.log | awk '{print $2}')
epochs=$(python3 -c "print(round(40 * 3843501 / (3843501 + $extra)))")
echo "cc3m states $extra, epochs $epochs"
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python scripts/tokae/train.py --dataset=imagenet --start_block=6 --end_block=12 \
    --views=3 --tokens=cls_patch_std --extra_dataset=cc3m --num_epochs=$epochs --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 \
    --jordan=True --top_k=256 --ae_dim=2049 --ae_hidden=2048 --lambda_embedding=1 --lambda_kl=0.03 \
    --name=im_cc3m_std_kl0.03 > /home/nsa325/work/koopmann/temp/im_cc3m_std_kl0.03.log 2>&1
echo "training exited with code $?"
