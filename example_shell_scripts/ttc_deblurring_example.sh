### This file is not intended to be used to run an experiment. It provides an example on how to train TTC critics for image denoise with ttc.py and how to evaluate the denoise performance using denosing_eval.py ###

#!/bin/bash

## Prepare virtual environment
source path/to/requirements-txt

## Settings
folder="./deblurring_results"
data="path/to/bsds500_dataset"
model="sndcgan"
net_dim=64
num_chan=3
height=128
width=128
step_info='(20)_1[1,1]'
gaussian_blur="5_2"
critters=2500
bs=32
generate_images='all_steps'

## Train
python train.py --folder=$folder --target=$target --source=$source_ --model=$model --net_dim=$net_dim --num_chan=$num_chan --height=$height --width=$width --step_info=$step_info --random_crop_source --random_crop_target --gaussian_blur=$gaussian_blur --critters=$critters --bs=$bs --compute_propagation_time --post_step_wasser_d --generate_images=$generate_images
