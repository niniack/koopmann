#!/bin/bash
# Input-ceiling probe on DINOv3 ViT-B/16: plain MLPs from the block-6 summary, no KAE (see ceiling_probe.py).
# Same data (ImageNet 3 views + the 21k images), standardization (of im_dino_w21_wd), optimizer, schedule and 12 epochs as im_dino_w21_wd.
SP=/home/nsa325/work/koopmann/temp
cd $SP
HF_HUB_OFFLINE=1 PYTHONPATH=$SP/tf_target $SP/venv/bin/python ceiling_probe.py --encoder=dinov3-vitb16 --reference=im_dino_w21_wd --name=probe_dino_b6 --num_epochs=12 --extra_dataset=w21
