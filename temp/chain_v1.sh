#!/bin/bash
until grep -q "training exited" /home/nsa325/work/koopmann/temp/run_std_outer.log; do sleep 30; done
/home/nsa325/work/koopmann/temp/run_im_jordan256_2k_patch_v1.sh > /home/nsa325/work/koopmann/temp/run_v1_outer.log 2>&1
