#!/bin/bash
cd /home/nsa325/work/koopmann/temp
HF_HUB_OFFLINE=1 exec ./venv/bin/python cls_train.py --name=cls --total_steps=30000
