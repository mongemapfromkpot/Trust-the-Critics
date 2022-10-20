# Trust-the-Critics

This repository is a PyTorch implementation of the TTC algorithm presented in *A new method for determining Wasserstein 1 optimal transport maps from Kantorovich potentials, with deep learning applications*. It also contains an implementation of the benchmark denoising method from Lunz et al. (2018) to which we compare the performance of TTC on denoising tasks.


## How to run this code ##
* Create a Python virtual environment with Python 3.8 installed.
* Install the packages listed in requirements.txt (pip install -r requirements.txt)

The various experiments we run with TTC are described in Section 5 and Addendix 8 of the paper. Illustrating its flexibility, TTC can be trained for all four applications in the paper -- i.e. image denoising, generation, translation and deblurring -- using the same train.py script; the only necessary changes are the source and target datasets, specified by the 'source' and 'target' arguments. New TTC steps can be added to a previous TTC run by specifying the same path with the 'folder' argument as for that previous run. We include samples of the shell scripts we used to run our experiments in the example_shell_scripts folder. Note that training TTC is computationally demanding, and thus requires adequate computational resources (i.e. running this on your laptop is not recommended).

### Computing architecture and running times
We ran the code presented here on computational clusters provided by the Digital Research Alliance of Cananda (https://alliancecan.ca/en), always using a single NVIDIA P100 or V100 GPU. Training times are reported in Addendix 8 of the paper.


## Assets 
Portions of this code, as well as the datasets used to produce our experimental results, make use of existing assets. We provide here a list of all assets used, along with the licenses under which they are distributed, if specified by the originator:
- **pytorch_fid**: from https://github.com/mseitzer/pytorch-fid. Distributed under the Apache License 2.0.
- **MNIST dataset**: from http://yann.lecun.com/exdb/mnist/. Distributed under the Creative Commons Attribution-Share Alike 3.0 license.
- **Fashion MNIST datset**: from  https://github.com/zalandoresearch/fashion-mnist ((c) 2017 Zalando SE, https://tech.zalando.com). Distributed under the MIT licence.
- **CelebA-HQ dataset**: from https://paperswithcode.com/dataset/celeba-hq
- **Image translation datasets**: from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix ((c) 2017, Jun-Yan Zhu and Taesung Park). Distributed under the BSD licence.
- **BSDS500 dataset**: from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html.
