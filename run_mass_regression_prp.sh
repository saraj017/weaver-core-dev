mkdir -p /fmhwwvol/experiments/mass_regression_test

python -u weaver/train.py \
 --gpu "0,1" \
 --train-mode regression \
 --batch-size 768 --start-lr 6.75e-3 --num-epochs 100 --optimizer ranger \
 --data-train '/fmhwwvol/ntuples/*/train/*.root' \
 --data-test '/fmhwwvol/ntuples/*/test/*.root' \
 --data-config weaver/data_new/finetune/FM_ak8_mass_regression.yaml \
 --network-config weaver/networks/fintune_test/mlp_2p_gated_regression.py \
 --model-prefix /fmhwwvol/experiments/mass_regression_test/model \
 --log-file /fmhwwvol/experiments/mass_regression_test/logs/train.log \
 --predict-output /fmhwwvol/experiments/mass_regression_test/predict/pred.root \
 --num-workers 0 \
 --in-memory \
 --steps-per-epoch=1 \
 --tensorboard _test \

mv runs/* /fmhwwvol/runs/