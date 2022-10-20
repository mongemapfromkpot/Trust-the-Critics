"""
Evaluating PSNR of TTC pretrained on a denoising application.
"""
import os
import sys
import shutil
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'ttc_utils'))
import time
import argparse
import logging
import pickle
import copy
import numpy as np
import pandas as pd
import random
import torch
from torchvision import transforms

import ttc_tools
import networks
import dataloaders
import generate_samples
import custom_transforms
from step_master import ttc_step




@torch.no_grad()
def psnr_calc(noisy, real):
    """Calculates PSNR between noisy and real data
    Inputs
    - noisy; batch of noisy data
    - real; batch of clean data, in correspondence with noisy
    Outputs
    - psnrs; full list of psnrs
    - mean and std of list of psnrs
    """
    numpix = noisy.size(1)*noisy.size(2)*noisy.size(3)
    bs = noisy.size(0)
    avg_sq_norm = (1/numpix)*torch.norm(0.5*(noisy.view(bs, -1) - real.view(bs,-1)), dim=1)**2 # Multiplication by 0.5 because vals between [-1,1]
    psnrs = -10*torch.log10(avg_sq_norm)
    return psnrs.cpu(), torch.tensor([torch.mean(psnrs), torch.std(psnrs)]).cpu()


def adv_reg(noisy_img, critic, num_step, mu, stepsize):
    """Implementation of adversarial regularization (benchmark denoiser), which solves a backward euler problem
    via gradient descent
    Inputs
    - noisy_img; batch of noisy images to be restored
    - critic; trained critic to be used as a learned regularizer
    - num_step; how many steps to do in gradient descent for restoration
    - mu; coefficient for learned regularizer. Computed from noise statistics
    - stepsize; stepsize for gradient descent algorithm
    Outputs
    - Restored batch of images
    """
    observation = noisy_img.detach().clone() # initial noisy observation
    for i in range(num_step):
        noisy_img = (ttc_step(noisy_img, critic, mu*stepsize) - 2*stepsize*(noisy_img-observation)).detach().clone()

    return noisy_img


@torch.no_grad()
def find_good_mu(noise_mag):
    """computes the regularization parameter in adversarial regularizer according to their heuristic"""
    return 2*torch.mean(noise_mag)




def denoise_eval(opts):
    """
    ...
    """
    # -------------------------- SETTING UP --------------------------
    message = "EVALUATING DENOISING PERFORMANCE (PSNR)\n\n"
    folder = opts.folder
    
    ### FOLDER(S)
    # Main folder
    required_files = ['experiment_log.txt', 'step_dict.pkl']
    if not ttc_tools.check_folder(folder, required_files):
        raise OSError("Given folder does not exist or does not contain all required files.")
    denoising_eval_folder = os.path.join(folder, 'denoising_evaluation_v2')
    os.makedirs(denoising_eval_folder)
    
    ### LOGGING
    log_file = os.path.join(folder, 'experiment_log.txt')
    
    logging.basicConfig(level = 0)
    logger = logging.getLogger(__name__) # We use a single logger with 2 handlers.
    logger.propagate = False
    logger.setLevel(0)
    
    console_handler = logging.StreamHandler() # Writes to console
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    console_handler.setLevel(1)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file) # Writes to log.txt
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    file_handler.setLevel(2)
    logger.addHandler(file_handler)
    
    
    ### DETERMINISTIC BEHAVIOR
    if opts.seed > 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark=False #If true, optimizes convolution for hardware, but gives non-deterministic behaviour
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)
        message += f"Random seed == {opts.seed}\n"
    else:
        torch.backends.cudnn.benchmark=True
        message += "Not using a random seed.\n"
    

    ### MAIN DEFINITIONS
    # Load step dict
    with open(os.path.join(folder, 'step_dict.pkl'), 'rb') as step_file:
        step_dict = pickle.load(step_file)
    crit_count = len(step_dict['type'])
    assert all(os.path.isfile(os.path.join(folder, 'critics', f'critic_{i+1}')) for i in range(crit_count))
    
    # Device(s)
    num_crit = len(step_dict['type'])
    num_gpus = opts.gpus if (opts.gpus > -1) else torch.cuda.device_count() # Maybe you don't want to use all the gpus available...
    if num_gpus > torch.cuda.device_count(): # Asked for more GPUs than are available.
        warning_message  = f"WARNING: {num_gpus} gpus demanded, but only {torch.cuda.device_count()} gpus available."
        warning_message += f"\nTraining will move forward using {torch.cuda.device_count()} gpus."
        logger.log(1, warning_message)
        num_gpus = torch.cuda.device_count()  
    device = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
    if not device: # device is empty, meaning num_gpus==0
        device.append(torch.device('cpu'))
    device = [device[(i*num_gpus)//num_crit] for i in range(num_crit)] # Now device[i] is the device for critic[i]
    
    # Data
    # Get a batch of images
    data_shape = step_dict['data_shape']
    num_samples = opts.num_samples
    loader = dataloaders.dataloader_tool(opts.target, num_samples, shape=data_shape, device=device[-1], train=False, random_crop=opts.random_crop, num_workers=opts.num_workers)
    load_it = iter(loader)
    clean_batch = next(load_it)[0]
    del loader, load_it
    # Get a noisy version
    noise_adder = custom_transforms.AddGaussianNoise(opts.noise_sigma)
    noisy_batch = noise_adder(clean_batch)
    clean_batch_ = clean_batch.to(device[-1])
    
    # Propagator
    propagator = ttc_tools.Propagator(step_dict)
    for i in range(crit_count):
        model, dim = step_dict['critic_params'][i][0], step_dict['critic_params'][i][1]
        critic = getattr(networks, model)(dim, data_shape).to(device[i])
        critic.load_state_dict(torch.load(os.path.join(folder, 'critics', f'critic_{i+1}'), map_location=device[i]))
        critic.to(device[i])
        critic.device = device[i]
        propagator.trained_critics.append(critic)

    # Initial message
    if opts.benchmark:
        message += "Will compute PSNR with both TTC and benchmark method "
    else:
        message += "Will compute PSNR with TTC "
    message += f"for {num_samples} samples.\n\n"
    logger.log(2, message)



    # -------------------------- DENOISING AND EVALUATING PSNRs --------------------------
    df_inds = list(range(num_samples))
    df_inds.extend(['mean', 'std'])
    
    # Get PSNR of noisy data
    logger.log(1, "Evaluating PSNRs for noisy data...\n")
    noisy_psnrs, noisy_mean_std = psnr_calc(noisy_batch, clean_batch)
    noisy_mean_std = noisy_mean_std.numpy()
    noisy_psnrs = np.append(noisy_psnrs.cpu().numpy(), noisy_mean_std)
    df = pd.DataFrame(noisy_psnrs, index=df_inds, columns=['Noisy'])
    
    # Denoising with TTC
    logger.log(1, "Denoising with TTC...")
    ttc_tools.add_bool_item(propagator.step_dict, opts.eval_steps, 'keep_samples_step', reset=True)
    ttc_denoised_batches = propagator.propagate_and_keep(noisy_batch.detach().clone())
    logger.log(1, "Evaluating PSNRs for TTC...\n")
    
    for i, ttc_denoised_batch in enumerate(ttc_denoised_batches):
        ttc_denoised_batch = ttc_denoised_batch.to(device[-1])
        ttc_psnrs, ttc_mean_std = psnr_calc(ttc_denoised_batch, clean_batch_)
        ttc_mean_std = ttc_mean_std.numpy()
        ttc_psnrs = np.append(ttc_psnrs.cpu().numpy(), ttc_mean_std)
        df[f'TTC_step_{i+1}'] = ttc_psnrs
    
    # Denoising with benchmark
    if opts.benchmark:
        logger.log(1, "Denoising with benchmark...")
        norm_diffs = torch.norm((clean_batch - noisy_batch).view(num_samples, -1), dim=1) 
        mu = find_good_mu(norm_diffs)
        noisy_batch_ = noisy_batch.to(device[0])
        bm_denoised_batch = adv_reg(noisy_batch_, propagator.trained_critics[0], opts.num_step_bm, mu, opts.stepsize_bm)
        logger.log(1, "Evaluating PSNRs for benchmark...\n")
        bm_psnrs, bm_mean_std = psnr_calc(bm_denoised_batch, clean_batch_)
        bm_mean_std = bm_mean_std.numpy()
        bm_psnrs = np.append(bm_psnrs.cpu().numpy(), bm_mean_std)
        df['Benchmark'] = bm_psnrs
    
    df.to_csv(os.path.join(denoising_eval_folder, 'psnrs.csv'), sep='\t')
    
    # Make images
    denoised_images_folder = os.path.join(denoising_eval_folder, 'images')
    os.makedirs(denoised_images_folder)
    
    for i, ttc_denoised_batch in enumerate(ttc_denoised_batches):
        step_images_folder = os.path.join(denoised_images_folder, f'step_{i+1}')
        os.makedirs(step_images_folder)
        batches_to_save = [clean_batch.detach().cpu(), 
                           noisy_batch.detach().cpu(),
                           bm_denoised_batch.detach().cpu(),
                           ttc_denoised_batch.detach().cpu()]
        grid_columns = opts.grid_columns if opts.grid_columns > 0 else 4
        generate_samples.batch_comparisons(step_images_folder, batches_to_save, grid_columns=grid_columns, ext='pdf', dpi=3000)
     
    # Compress
    shutil.make_archive(os.path.join(denoising_eval_folder, 'denoised_images'), 'zip', denoised_images_folder)
    shutil.rmtree(denoised_images_folder)
    
    


    
    
    
    # -------------------------- EPILOGUE --------------------------
    
    message = f"PSNRs obtained with TTC (mean +- std):    {ttc_mean_std[0]:.2f} +- {ttc_mean_std[1]:.2f}\n"
    if opts.benchmark:
        message += f"PSNRs obtained with BENCHMARK (mean +- std):    {bm_mean_std[0]:.2f} +- {bm_mean_std[1]:.2f}"
    
    message +=  "\n\n\n\n----------------------\n\n\n"
    logger.log(2, message)
    
    
    
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('PSNR evaluation for TTC and benchmark denoising algorithm')
    
    # Paths
    parser.add_argument('--folder', type=str, required=True,   help="Folder to create or from which to resume")
    parser.add_argument('--target', type=str, required=True,   help="Directory where target data is located")
    
    # PSNR evaluation settings
    parser.add_argument('--eval_steps', default='last_step', choices=['all_steps', 'all_critics', 'last_step'], help="Specifies after which steps to evaluate PSNR.")
    parser.add_argument('--num_samples',  type=int, default=200, help='Number of individual images on which PSNR is evaluated.')
    parser.add_argument('--grid_columns', type=int, default=0,   help='If > 0, will save a grid image containing all test denoising examples.')
    
    # Data transforms
    parser.add_argument('--random_crop',   action='store_true',    help="If true, will reshape data through random cropping instead of resizing.")
    parser.add_argument('--noise_sigma',   type=float, default=0., help="Standard deviation of noise to add to source data. Use for denoising experiments.")
    
    # Optional task
    parser.add_argument('--benchmark',   action='store_true',      help="If true, will perform denoising with benchmark method as well.")
    parser.add_argument('--num_step_bm', type=int,   default=200,  help="Number of steps for benchmark (adversarial regularization)")
    parser.add_argument('--stepsize_bm', type=float, default=0.05, help="Gradient descent step size for benchmark (adversarial regularization)")
    
    # Other
    parser.add_argument('--seed',        type=int,   default=-1, help="Set random seed for reproducibility.")
    parser.add_argument('--gpus',        type=int,   default=-1, help="Number of GPUs to use. Each critic will act on a single GPU, but different critics may be on different GPUs.")
    parser.add_argument('--num_workers', type=int,   default=0,  help="Number of data loader processes.")
    
    opts = parser.parse_args()
    denoise_eval(opts)