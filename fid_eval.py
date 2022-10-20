"""
Evaluating FID of pretrained TTC
"""

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'ttc_utils'))
import re
import time
from tqdm import tqdm
import argparse
import numpy as np
import random
import pandas as pd
import logging
import pickle
import torch
from torchvision import transforms
from pytorch_fid import fid_score
from pytorch_fid import inception

import ttc_tools
import generate_samples
import networks
import dataloaders
import custom_transforms



def fid_eval(opts):

    # -------------------------- SETTING UP --------------------------
    message = "EVALUATING FID\n\n"
    folder = opts.folder
    
    ### FOLDER(S)
    # Main folder
    required_files = ['experiment_log.txt', 'step_dict.pkl', 'results.csv']
    if not ttc_tools.check_folder(folder, required_files):
        raise OSError("Given folder does not exist or does not contain all required files.")
    compute_target_stats = (opts.target.split('.')[-1] != 'npz') # If target ends with npz, we assume it contains the inception statistics
                                                                 # of the target, so we avoid computing those again                                                                                           
    # Folder where to save generated data before FID evaluation
    temp_folder = opts.temp_folder if (opts.temp_folder != '') else folder
    gen_data_folder = os.path.join(temp_folder, 'generated_data')
    os.makedirs(gen_data_folder, exist_ok=False)
    if compute_target_stats:
        target_ = os.path.join(temp_folder, 'target_data')
        os.makedirs(target_, exist_ok=False)
    else:
        target_ = opts.target
    
    
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
    
    # Dataloaders
    data_shape = step_dict['data_shape']
    ## Source
    pretransforms = []
    if opts.noise_sigma:
        pretransforms.append(custom_transforms.AddGaussianNoise(opts.noise_sigma))
    if opts.gaussian_blur: # opts.gaussian_blur != ''
        blur_params = [int(p) for p in opts.gaussian_blur.split('_')]
        ks = blur_params[0]
        if len(blur_params) == 2: #  opts.gaussian_blur = 'kernelsize_sigma'
            pretransforms.append(transforms.GaussianBlur(kernel_size=ks, sigma=blur_params[1]))
        elif len(blur_params) == 3:  #  opts.gaussian_blur = 'kernelsize_sigmamin_sigmamax'
            pretransforms.append(transforms.GaussianBlur(kernel_size=ks, sigma=(blur_params[1], blur_params[2])))
        else:
            raise ValueError('Invalid gaussian_blur argument.')
    source_loader = dataloaders.dataloader_tool(opts.source, opts.bs, shape=data_shape, pretransforms=pretransforms, device=device[0],  train=False, random_crop=opts.random_crop_source, num_workers=opts.num_workers)
    ## Target
    if compute_target_stats:
        target_loader = dataloaders.dataloader_tool(opts.target, opts.bs, shape=data_shape, device=device[-1], train=False, random_crop=opts.random_crop_target, num_workers=opts.num_workers)
    num_batch = opts.num_samples//opts.bs
    
    # Propagator
    propagator = ttc_tools.Propagator(step_dict)
    for i in range(crit_count):
        model, dim = step_dict['critic_params'][i][0], step_dict['critic_params'][i][1]
        critic = getattr(networks, model)(dim, data_shape).to(device[i])
        critic.load_state_dict(torch.load(os.path.join(folder, 'critics', f'critic_{i+1}'), map_location=device[i]))
        critic.to(device[i])
        critic.device = device[i]
        propagator.trained_critics.append(critic)
    
    # Results dataframe
    results_df = pd.read_csv(os.path.join(folder, 'results.csv'), sep=',', index_col=['Critic', 'Step'])
    if ('FID' not in results_df.columns):
        results_df['FID'] = np.nan
    
    # Add evaluation steps to step_dict
    eval_steps = opts.eval_steps
    ttc_tools.add_bool_item(propagator.step_dict, eval_steps, 'fid_step')
    if eval_steps == 'all_steps':
        message += "Will evaluate FID after each step of each critic.\n"
    elif eval_steps == 'all_critics':
        message += "Will evaluate FID after the last step of each critic.\n"
    elif eval_steps == "last_step":
        message += "Will evaluate FID only after full propagation, i.e. after the last step of the last critic.\n"
        
    # Save steps_dict
    with open(os.path.join(folder, 'step_dict.pkl'), 'wb') as f:
        pickle.dump(propagator.step_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Initial message
    logger.log(2, message)
        


    # -------------------------- GENERATE DATA  --------------------------
    # Generate data with TTC
    logger.log(1, "Generating data from source with TTC...")
    s_iter = iter(source_loader)   
    for i in tqdm(range(num_batch)):
        in_batch, s_iter = ttc_tools.get_data(s_iter, source_loader)
        save_source = (('source',0) not in results_df.index)
        propagator.propagate_and_save(gen_data_folder, 'fid_step', in_batch, batch_idx=i, save_source=save_source, to_rgb=True)
    logger.log(1, "Done.\n")
    del propagator # We don't need the critics anymore, so we remove them to free up memory space.

    # Put target data in a folder
    num_samples_target = opts.num_samples_target if (opts.num_samples_target > 0) else float('inf')
    if compute_target_stats:
        t_iter = iter(target_loader)
        for i, (batch, _) in enumerate(t_iter):
            generate_samples.save_individuals(target_, batch, batch_idx=i, to_rgb=True)
            if (i+1) * opts.bs >= num_samples_target:
                break



    # -------------------------- COMPUTE FID --------------------------
    # Prepare inception network
    block_idx = inception.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_net = inception.InceptionV3([block_idx]).to(device[0])
    
    # Start by computing inception statistics of real data. Save for later use.
    if compute_target_stats:
        logger.log(1, "Computing target data inception statistics...")
        start_time = time.perf_counter()
    target_mean, target_cov = fid_score.compute_statistics_of_path(target_, inception_net, 100, 2048, device[0]) # if target is an .npz file containing the stats, no computation is performed.
    if compute_target_stats:
        np.savez(os.path.join(folder, 'target_data_inception_stats.npz'), mu=target_mean, sigma=target_cov) 
        logger.log(1, f"Done in {(time.perf_counter() - start_time):.01f} seconds.\n")
    else:
        logger.log(1, "Target data inception statistics pre-computed.\n")
    
    # Compute inception statistics and FID for each evaluation step
    logger.log(1, "Computing inception statistics of generated data at each evaluation step...")
    start_time = time.perf_counter()
    for step_folder in os.listdir(gen_data_folder):
        time_ = time.perf_counter()
        gen_mean, gen_cov = fid_score.compute_statistics_of_path(os.path.join(gen_data_folder, step_folder), inception_net, 100, 2048, device[0])
        fid = fid_score.calculate_frechet_distance(gen_mean, gen_cov, target_mean, target_cov)
        # Get dataframe index from name of folder
        if step_folder=='source':
            results_df.loc[(0,0), :] = np.nan
            results_df.loc[(0,0), 'FID'] = fid
            results_df.sort_index(inplace=True)
            logger.log(1, f"FID of source == {fid}  \t(Evaluated in {(time.perf_counter() - time_):.01f} seconds)")    
        else:
            num_steps, num_crit = [int(x) for x in re.findall(r'[0-9]+', step_folder)]
            results_df.FID[(num_crit, num_steps)] = fid
            logger.log(1, f"After step {num_steps} (critics number {num_crit}):  FID = {fid}  \t(Evaluated in {(time.perf_counter() - time_):.01f} seconds)")
    logger.log(1, "Done.\n")
        


    # -------------------------- EPILOGUE --------------------------
    results_df.to_csv(os.path.join(folder, 'results.csv'), sep=',')
    
    # Best FID
    best_crit, best_step = results_df.FID.idxmin()
    best_FID = results_df.FID.loc[(best_crit, best_step)]
    
    # Write summary in experiment_log.txt
    message  = f"FULL RESULTS obtained up to now: \n{results_df}\n"
    message += f"\nBest FID up to now = {best_FID}, obtained after taking step number {best_step} of critic number {best_crit}."
    message +=  "\n\n\n\n----------------------\n\n\n"
    logger.log(2, message)
    
    





if __name__ == '__main__':
       
    parser = argparse.ArgumentParser('FID evaluation for TTC')
    
    # Paths
    parser.add_argument('--folder',      type=str, required=True,   help="Main folder created by running train.py")
    parser.add_argument('--temp_folder', type=str, default='',      help="Folder to temporarily save samples for FID evaluation")
    parser.add_argument('--target',      type=str, required=True,   help="Directory where target data is located")
    parser.add_argument('--source',      type=str, default='noise', help="Either noise (for Gaussian noise source) or directory where source data is located.")
    
    # FID evaluation settings
    parser.add_argument('--eval_steps', default='all_steps', choices=['all_steps', 'all_critics', 'last_step'], help="Specifies after which steps to evaluate FID.")
    parser.add_argument('--num_samples',        type=int, default=10000, help='Number of samples generated with TTC for FID')
    parser.add_argument('--num_samples_target', type=int, default=-1,    help='Number of samples from target distribution to use. -1 means all samples available will be used.')
    parser.add_argument('--bs',                 type=int, default=128,   help="Batch size used when training critics.")
    parser.add_argument('--recompute_steps',    action='store_true',     help="If True, will recompute FID for steps where it was previously computed and steps that were ignored in previous FID evaluation.")
    
    # Data transforms
    parser.add_argument('--random_crop_target', action='store_true', help="If true, will reshape target data through random cropping instead of resizing.")
    parser.add_argument('--random_crop_source', action='store_true', help="If true, will reshape source data through random cropping instead of resizing.")
    parser.add_argument('--noise_sigma',   type=float, default=0.,   help="Standard deviation of noise to add to source data. Use for denoising experiments.")
    parser.add_argument('--gaussian_blur', type=str, default='',     help="Parameters for blurring filter. Use for deblurring experiments. Specify blurring parameters as 'kernelsize_minblur_maxblur'.")
    
    # Other
    parser.add_argument('--seed',        type=int, default=-1, help="Set random seed for reproducibility.")
    parser.add_argument('--gpus',        type=int, default=-1, help="Number of GPUs to use. Each critic will act on a single GPU, but different critics may be on different GPUs.")
    parser.add_argument('--num_workers', type=int, default=0,  help="Number of data loader processes.")

    opts = parser.parse_args()
    fid_eval(opts)