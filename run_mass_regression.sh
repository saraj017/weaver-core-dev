#!/bin/bash

python -u weaver/train.py \
 --gpu "0,1" \
 --train-mode regression \
 --batch-size 768 --start-lr 6.75e-4 --num-epochs 3000 --optimizer ranger \
 --data-train '/fmhwwvol/ntuples/*/train/*.root' \
 --data-test '/fmhwwvol/ntuples/*/test/*.root' \
 --data-config weaver/data_new/finetune/FM_ak8_mass_regression_v10.yaml \
 --network-config weaver/networks/fintune_test/mlp_2p_gated_regression.py \
 --model-prefix /fmhwwvol/experiments/mass_regression_v10/model \
 --log-file /fmhwwvol/experiments/mass_regression_v10/logs/train.log \
 --predict-output /fmhwwvol/experiments/mass_regression_v10/predict/pred.root \
 --num-workers 0 \
 --in-memory \
 --steps-per-epoch=1 \
 --tensorboard _v10 \
 --load-model-weights finetune_gghww_custom \

mv runs/* /fmhwwvol/runs/