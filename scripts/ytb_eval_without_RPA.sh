config="configs.resnet101_rpcm_ytb_stage_1"

# only use propagation-correction modulator

# eval on YTB18
datasets="youtubevos18"
python ../tools/eval.py --config ${config} --dataset ${datasets} --ckpt_step 400000 --global_chunks 4 --gpu_id 1 

# eval on YTB19
datasets="youtubevos19"
python ../tools/eval.py --config ${config} --dataset ${datasets} --ckpt_step 400000 --global_chunks 4 --gpu_id 1 
