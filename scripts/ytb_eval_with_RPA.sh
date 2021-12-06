config="configs.resnet101_p2t"

# using both propagation-correction modulator and reliable proxy augmentation

# eval YTB19
datasets="youtubevos19"
python ../tools/eval_rpa.py --config ${config} --dataset ${datasets} --ckpt_step 400000 --global_chunks 4 --gpu_id 0 

# eval YTB19
datasets="youtubevos18"
python ../tools/eval_rpa.py --config ${config} --dataset ${datasets} --ckpt_step 400000 --global_chunks 4 --gpu_id 0 
