### This file is not intended to be used to run an experiment. It only provides an example of how to train TTC for image generation on MNIST and how to evaluate the generative performance with FID using this codebase. ###

#!/bin/bash

## Prepare virtual environment
source path/to/requirements-txt

## Settings
folder="./mnist_generation_results"
temp_folder=$temp_folder
target="path/to/MNIST"
model='infogan'
net_dim=64
num_chan=1
height=32
width=32
step_info='(10)_2[1,1]__:__(20)_1[1,1]'
critters=1000
bs=128
generate_images='all_steps'
eval_steps='all_steps'

## Train
python train.py --folder=$folder --target=$target --source='noise' --model=$model --net_dim=$net_dim --num_chan=$num_chan --height=$height --width=$width --step_info=$step_info --critters=$critters --bs=$bs --compute_propagation_time --post_step_wasser_d --generate_images=$generate_images

## Evaluate FID
python FID_eval.py --folder=$folder --temp_folder=$temp_folder --target=$target --num_samples=$num_samples_fid --bs=$bs
