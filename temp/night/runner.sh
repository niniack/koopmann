#!/bin/bash
# Overnight GPU queue: waits for the running job, then runs night/queue.txt one line at a time (name|extra args).
# Lines can be added, removed or reordered while it runs; it always takes the first line.
SP=/home/nsa325/work/koopmann/temp
Q=$SP/night/queue.txt
COMMON="--dataset=imagenet --start_block=6 --end_block=12 --views=3 --num_epochs=8 --lr=0.0075 --adam_lr=0.00075 --lambda_variance=0 --lambda_covariance=0 --lambda_isometric=0.1 --ae_dim=2049 --ae_hidden=4096 --ae_layers=2 --lambda_embedding=1 --lambda_kl=0.03"
while kill -0 3438526 2>/dev/null; do sleep 30; done
echo "$(TZ=America/New_York date +%H:%M) previous job done"
cd /home/nsa325/work/koopmann
while [ -s "$Q" ]; do
    line=$(head -1 "$Q"); sed -i 1d "$Q"
    [ "${line:0:1}" = "#" ] && continue
    name=${line%%|*}; args=${line#*|}
    if [ "${args:0:6}" = "SHELL:" ]; then
        echo "$(TZ=America/New_York date +%H:%M) start $name (shell)"
        bash -c "${args:6}" > $SP/night/$name.log 2>&1
    else
        echo "$(TZ=America/New_York date +%H:%M) start $name: $args"
        HF_HUB_OFFLINE=1 $SP/venv/bin/python scripts/tokae/train.py $COMMON $args --name=$name > $SP/night/$name.log 2>&1
    fi
    echo "$(TZ=America/New_York date +%H:%M) $name exited with code $?"
done
echo "$(TZ=America/New_York date +%H:%M) queue empty"
