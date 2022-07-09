config="configs.resnet101_rpcm_ytb_stage_1"
datasets="youtubevos"
# first stage training
python ../tools/train.py --config ${config} --datasets ${datasets}  --global_chunks 1

# second stage training
config="configs.resnet101_rpcm_ytb_stage_2"
python ../tools/train.py --config ${config} --datasets ${datasets}  --global_chunks 1


