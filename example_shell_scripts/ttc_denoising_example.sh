### This file is not intended to be used to run an experiment. It provides an example on how to train TTC critics for image denoise with ttc.py and how to evaluate the denoise performance using denosing_eval.py ###

#!/bin/bash

## Prepare virtual environment
source path/to/requirements-txt

## Settings
folder="./denoising_results"
data_path="path/to/bsds500_dataset"
model="arConvNet"
net_dim=64
num_chan=3
height=128
width=128
step_info='(20)_1[1,1]'
noise_sigma=0.2 
critters=2500
bs=32
generate_images='all_steps'
eval_steps='all_steps'


## Train
python train.py --folder=$folder --target=$data_path --source=$data_path --model=$model --net_dim=$net_dim --num_chan=$num_chan --height=$height --width=$width --step_info=$step_info --random_crop_source --random_crop_target --noise_sigma=$noise_sigma --critters=$critters --bs=$bs --compute_propagation_time --post_step_wasser_d --generate_images=$generate_images


## Evaluate PSNRs
python denoise_eval_v2.py --folder=$folder --target=$data_path --random_crop --noise_sigma=$noise_sigma --num_samples=$num_samples_eval --benchmark
