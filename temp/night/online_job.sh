#!/bin/bash
# Online-crop KAEs with and without a patch-token control (see control_train.py): plain, additive control, bilinear control.
cd /home/nsa325/work/koopmann/temp
HF_HUB_OFFLINE=1 /home/nsa325/work/koopmann/temp/venv/bin/python control_train.py --name=on --controls=none,add,bilinear --total_steps=30000
