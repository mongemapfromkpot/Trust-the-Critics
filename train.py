"""
Main training function for TTC
"""
import os
import sys
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
import stats_tracker
import custom_transforms





def train(opts):

    # -------------------------- SETTING UP --------------------------
    message = "TRAINING TTC\n\n"
    folder = opts.folder
    
    ### FOLDER
    required_files = ['experiment_log.txt', 'training_stats.pkl', 'step_dict.pkl']
    resuming = ttc_tools.check_folder(folder, required_files)
    os.makedirs(os.path.join(folder, 'critics'), exist_ok=True)
    

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
    
    file_handler = logging.FileHandler(log_file) # Writes to experiment_log.txt
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
    # TTC steps
    if opts.step_info == '':
        step_info = f"({opts.num_crit})_{opts.steps_per_crit}[{opts.step_type},{opts.theta}]"
    else:
        step_info = opts.step_info
    default_step_params = [opts.step_type, opts.theta]
    step_dict = ttc_tools.unpack_step_info(step_info, default_step_params)
    data_shape = [opts.num_chan, opts.height, opts.width]
    if resuming:
        # Load previous run
        stats_tracker.load(folder)
        with open(os.path.join(folder, 'step_dict.pkl'), 'rb') as step_file:
            previous_step_dict = pickle.load(step_file)
        step_dict = ttc_tools.glue_step_dicts(previous_step_dict, step_dict)
        data_shape = step_dict['data_shape']
        num_crit = len(step_dict['theta'])
        num_pretrained_crit = len(previous_step_dict['type'])
        num_previous_steps = sum([len(step_list) for step_list in previous_step_dict['type']])
        # Check the previous critics are correctly saved
        assert all(os.path.isfile(os.path.join(folder, 'critics', f'critic_{i+1}')) for i in range(num_pretrained_crit))
        message += f"Resuming from previous run save in {folder}.\n{num_pretrained_crit} pretrained critic(s) - will train {num_crit - num_pretrained_crit} new critics.\n"
    else:
        num_pretrained_crit = 0
        num_previous_steps = 0
        num_crit = len(step_dict['theta'])
        message += f"Training from scratch - will train {num_crit} critics.\n"
    step_dict['critic_params'] += [(opts.model, opts.net_dim)] * (num_crit - num_pretrained_crit)
    
    # Device(s)
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
    
    # Dataloaders, loss function parameters
    # Target
    target_loader = dataloaders.dataloader_tool(opts.target, opts.bs, shape=data_shape, device=device[-1], train=True, random_crop=opts.random_crop_target, num_workers=opts.num_workers)
    data_shape = target_loader.shape
    step_dict['data_shape'] = data_shape # If resuming, this stays as before.
    
    # Source transforms
    pretransforms = []
    if opts.noise_sigma: # opts.noise_sigma != 0
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
    # Source
    source_loader = dataloaders.dataloader_tool(opts.source, opts.bs, shape=data_shape, pretransforms=pretransforms, device=device[0],  train=True, random_crop=opts.random_crop_source, num_workers=opts.num_workers)
    loss_params = {'lr':opts.lr, 'betas':(opts.beta_1, opts.beta_2), 'weight_decay':opts.weight_decay, 'lambda':opts.lamb}
    message += f"Data shape [num_chan, height, width]: {data_shape}.\n"
    if opts.random_crop_source:
        message += "If source data needs reshaping, it will be done through random cropping.\n"
    if opts.random_crop_target:
        message += "If target data needs reshaping, it will be done through random cropping.\n"
    message += f"Network model: {opts.model}. \tNetwork dimension parameter: {opts.net_dim}.\n"
    message += f"Number of training iterations per critic: {opts.critters}. \tBatch size: {opts.bs}.\n"
    message += f"Loss function and optimization parameters: {loss_params}.\n"
    if opts.noise_sigma:
        message += f"Gaussian noise is added to the source data, with a standard deviation of {opts.noise_sigma}.\n"
    if opts.gaussian_blur:
        message += f"Gaussian blurring source images, with kernel size = {ks} "
        if len(blur_params) == 2:
            message += f"and constant standard deviation = {blur_params[1]}.\n"
        else:
            message += f"and standard deviation picked uniformly at random between {blur_params[1]} and {blur_params[2]}.\n"

    # Propagator, critic architecture
    propagator = ttc_tools.Propagator(step_dict)
    if resuming:
        for i in range(num_pretrained_crit):
            model, dim = step_dict['critic_params'][i][0], step_dict['critic_params'][i][1]
            critic = getattr(networks, model)(dim, data_shape).to(device[i])
            critic.load_state_dict(torch.load(os.path.join(folder, 'critics', f'critic_{i+1}'), map_location=device[i]))
            critic.device = device[i]
            propagator.trained_critics.append(critic)
    logger.log(2, message)
    
    
    
    # -------------------------- TRAINING --------------------------
    # To record results
    df_indices = [(0,num_previous_steps)]
    wasser_d = []
    if opts.post_step_wasser_d:
        wasser_d_ps = [] # ps stands for 'post step'
    
    start_time = time.perf_counter()
    for i in range(num_pretrained_crit, num_crit):
        # Initialize a new critic (at previous critic)
        critic = getattr(networks, opts.model)(opts.net_dim, data_shape)
        critic.device = device[i]
        if i > 0:
            critic.load_state_dict(propagator.trained_critics[-1].state_dict())
        critic.to(device[i])
        
        # Train new critic, add to propagator, save 
        logger.log(1, f"\nCRITIC {i+1}.\nTraining...")
        time_ = time.perf_counter()
        ttc_tools.critic_trainer(critic, opts.critters, source_loader, target_loader, propagator, loss_params)
        critic_training_time = time.perf_counter() - time_
        propagator.trained_critics.append(critic)
        stats_tracker.plot('critic_index', int(i+1))
        stats_tracker.plot('critic_training_time', critic_training_time)
        logger.log(1, f"Done in {critic_training_time:.01f} seconds \t ({(critic_training_time/opts.critters):.6f} sec/batch)\n")
        torch.save(copy.deepcopy(critic).to(torch.device('cpu')).state_dict(), os.path.join(folder, 'critics', f'critic_{i+1}'))

        # Compute Wasserstein distance, fill in step size information in propagator
        message  = f"Computing estimated Wasserstein distance(s) for each step  ({len(propagator.step_dict['theta'][i])} step(s) for this critic)..."
        logger.log(1, message)
        new_wasser_d = []
        results_message = ""
        for j, th in enumerate(propagator.step_dict['theta'][i]):
            wasserstein_d = ttc_tools.get_wasserstein_D(critic, 5, source_loader, target_loader, propagator, loss_params)
            propagator.step_dict['step_size'][i].append(th * wasserstein_d)
            df_indices.append((i+1, df_indices[-1][1]+1))
            new_wasser_d.append(wasserstein_d)
            results_message += f"Before taking step {j+1}:  Estimated Wasserstein distance = {wasserstein_d:.5f}.\n"
        wasser_d.extend(new_wasser_d)
        if opts.post_step_wasser_d:
            logger.log(1, "Computing extra Wasserstein D after last step...")
            wasserstein_d = ttc_tools.get_wasserstein_D(critic, 5, source_loader, target_loader, propagator, loss_params)
            results_message += f"After taking step {len(propagator.step_dict['theta'][i])}:  Estimated Wasserstein distance = {wasserstein_d:.5f}.\n"
            new_wasser_d_ps = new_wasser_d[1:] + [wasserstein_d]
            wasser_d_ps.extend(new_wasser_d_ps)
        # Save steps_dict
        with open(os.path.join(folder, 'step_dict.pkl'), 'wb') as f:
            pickle.dump(propagator.step_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        message = "Done. Results:\n" + results_message 
        logger.log(1, message)
        
        # Propagation times
        if opts.compute_propagation_time:
            s_iter = iter(source_loader)
            time_ = time.perf_counter()
            for j in range(10):
                batch, s_iter = ttc_tools.get_data(s_iter, source_loader)
                _ = propagator.propagate_data(batch)
            prop_time_batch = (time.perf_counter() - time_)/50
            stats_tracker.plot('propagation_time_per_batch', prop_time_batch)
            logger.log(1, f"Propagation time up to and including critic {i+1}: {prop_time_batch} sec/batch.\n")

        stats_tracker.tick()
        stats_tracker.flush(folder)
        
        

    # -------------------------- EPILOGUE --------------------------
    # Save results in a dataframe    
    df_indices = pd.MultiIndex.from_tuples(df_indices[1:], names=['Critic', 'Step'])
    df_data = np.asarray([wasser_d, wasser_d_ps]).transpose() if opts.post_step_wasser_d else np.asarray(wasser_d)
    columns = ['WD_pre-step', 'WD_post-step'] if opts.post_step_wasser_d else ['WD_pre-step']
    results_df = pd.DataFrame(df_data, index=df_indices, columns=columns).sort_index()
    if 'results.csv' in os.listdir(folder):
        previous_results = pd.read_csv(os.path.join(folder, 'results.csv'), sep=',', index_col=['Critic', 'Step'])
        results_df = pd.concat([previous_results, results_df])
    results_df.to_csv(os.path.join(folder, 'results.csv'), sep=',')
    
    # Generate images
    if opts.generate_images:
        gen_images_path = os.path.join(folder, 'generated_images')
        os.makedirs(gen_images_path, exist_ok=True)
        ttc_tools.add_bool_item(propagator.step_dict, opts.generate_images, 'batch_image', reset=True)
        s_iter = iter(source_loader)
        batch, _ = next(s_iter)
        if batch.size(0) > 64:
            batch = batch[:64]
        propagator.propagate_and_save(gen_images_path, 'batch_image', batch, batch_image=True, save_source=True)
        
    # Save steps_dict
    with open(os.path.join(folder, 'step_dict.pkl'), 'wb') as f:
        pickle.dump(propagator.step_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Write summary to experiment log
    total_time = time.perf_counter() - start_time
    message  = f"\nDone after {int(total_time)} seconds.\n"
    message += f"\nFULL RESULTS obtained up to now: \n{results_df.__repr__()}"
    message +=  "\n\n\n\n----------------------\n\n\n"
    logger.log(2, message)





if __name__ == '__main__':
        
    parser = argparse.ArgumentParser('TTC training')
    
    # Paths
    parser.add_argument('--folder', type=str, required=True,   help="Folder to create or from which to resume")
    parser.add_argument('--target', type=str, required=True,   help="Directory where target data is located")
    parser.add_argument('--source', type=str, default='noise', help="Either noise (for Gaussian noise source) or directory where source data is located.")
    
    # Model settings
    parser.add_argument('--model',    default='infogan', choices=['dcgan', 'infogan', 'arConvNet', 'sndcgan','bsndcgan', 'norm_taker'], help="Architecture for critics to be trained")
    parser.add_argument('--net_dim',  type=int, default=64, help="Number determining critic network dimensions")
    parser.add_argument('--num_chan', type=int, default=-1, help='Number of color channels. If =-1, gets inferred from data.')
    parser.add_argument('--height',   type=int, default=-1, help='Image height in pixels. If =-1, gets inferred from data.')
    parser.add_argument('--width',    type=int, default=-1, help='Image width in pixels. If =-1, gets inferred from data.')
    
    # Step information
    parser.add_argument('--num_crit',       type=int,   default=1,  choices=range(1,100), help="Number of critics to train for iterative TTC.")
    parser.add_argument('--steps_per_crit', type=int,   default=1,  help="Number of TTC steps taken on each critic.")
    parser.add_argument('--step_type',      type=int,   default=1,  help="Determines type of step to use. See step_master.py")
    parser.add_argument('--theta',          type=float, default=1,  help="Fraction of W1 distance giving the step size.")
    parser.add_argument('--step_info',      type=str,   default='', help="Information about steps taken on each critic. See description in  \
                                                                            step_tools.py Specifying any non-empty argument will override the \
                                                                            num_crit, steps_per_crit, step_type and theta arguments.")
    # Training settings 
    parser.add_argument('--critters',     type=int,   default=1000,  help="Number of training iterations used to train each critic.")
    parser.add_argument('--bs',           type=int,   default=128,   help="Batch size used when training critics.")
    parser.add_argument('--lamb',         type=float, default=1000., help="Gradient penalty multiplier in loss function.")
    parser.add_argument('--lr',           type=float, default=1e-4,  help="Learning rate.")
    parser.add_argument('--beta_1',       type=float, default=0.5,   help="beta_1 parameter for Adam optimizer.")
    parser.add_argument('--beta_2',       type=float, default=0.999, help="beta_2 parameter for Adam optimizer")
    parser.add_argument('--weight_decay', type=float, default=0.,    help="Weight decay parameter for Adam optimizer.")
    
    # Data transforms
    parser.add_argument('--random_crop_target',   action='store_true', help="If true, will reshape target data through random cropping instead of resizing.")
    parser.add_argument('--random_crop_source',   action='store_true', help="If true, will reshape source data through random cropping instead of resizing.")
    parser.add_argument('--noise_sigma',   type=float, default=0.,     help="Standard deviation of noise to add to source data. Use for denoising experiments.")
    parser.add_argument('--gaussian_blur', type=str, default='',       help="Parameters for blurring filter. Use for deblurring experiments. Specify blurring parameters as 'kernelsize_minblur_maxblur'.")
    
    # Optional tasks
    parser.add_argument('--compute_propagation_time', action='store_true', help="If true, time to propagate data through TTC will be recorded.")
    parser.add_argument('--post_step_wasser_d',       action='store_true', help="Whether to compute Wasserstein distance estimated by each critic after the critic's last step.")
    parser.add_argument('--generate_images', default='all_steps', choices=['all_steps', 'all_critics', 'last_step', ''], help="Specifies wether to generate images after training.")
    
    # Other
    parser.add_argument('--seed',        type=int, default=-1, help="Set random seed for reproducibility.")
    parser.add_argument('--gpus',        type=int, default=-1, help="Number of GPUs to use. Each critic will act on a single GPU, but different critics may be on different GPUs.")
    parser.add_argument('--num_workers', type=int, default=0,  help="Number of data loader processes.")
    
    opts = parser.parse_args()
    train(opts)