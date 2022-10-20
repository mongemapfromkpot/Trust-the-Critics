### This file is not intended to be used to run an experiment. It provides an example of how to train TTC for image translation ###

#!/bin/bash

## Prepare virtual environment
source path/to/requirements.txt

## Settings
folder="./translation_results"
target="path/to/monet_dataset"
source_="path/to/photo_dataset"
model="sndcgan"
net_dim=64
num_chan=3
height=128
width=128
step_info='(20)_1[1,1]'
critters=2500
bs=32
lr=0.0001
generate_images='all_steps'

## Train
python train.py --folder=$folder --target=$target --source=$source_ --model=$model --net_dim=$net_dim --num_chan=$num_chan --height=$height --width=$width --step_info=$step_info --random_crop_target --critters=$critters --bs=$bs --lr=$lr --compute_propagation_time --post_step_wasser_d --generate_images=$generate_images
