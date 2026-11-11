#!/bin/bash
# Input-ceiling probe on CLIP ViT-B/32: plain MLPs from the block-6 summary, no KAE (see ceiling_probe.py).
# Same data, standardization (of im_iso_2x4k_wd), optimizer, schedule and 8 epochs as the CLIP pilots.
SP=/home/nsa325/work/koopmann/temp
cd $SP
HF_HUB_OFFLINE=1 PYTHONPATH=$SP/tf_target $SP/venv/bin/python ceiling_probe.py --encoder=ViT-B-32 --reference=im_iso_2x4k_wd --name=probe_clip_b6 --num_epochs=8
