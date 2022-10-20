import os
import re
from tqdm import tqdm
import torch
from torch import autograd, optim

import step_master
import stats_tracker
import generate_samples


# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
"""
TTC data propagtion and training.
"""

def get_data(generic_iterator, generic_loader):
    """
    Gets minibatch from data iterator
    Inputs:
    - generic_iterator; iterator for dataset
    - generic_loader; loader for dataset
    Outputs:
    - data; minibatch of data from iterator
    - generic_iterator; iterator for dataset, reset if
    you've reached the end of the dataset
    """
    try:
        data = next(generic_iterator)[0]
    
    except StopIteration:
        generic_iterator = iter(generic_loader)
        data = next(generic_iterator)[0]

    return data, generic_iterator



def calc_grad_penalty(critic, s_batch, t_batch, lamb, create_graph=True):
    """
    Computes and returns a gradient penalty for 'critic'. The gradient penalty is applied at 
    interpolation points drawn uniformly on segments between the elements of the source batch 
    's_batch' and the target batch 't_batch'. The 'lamb' (for lambda) parameter is a multiplier.
    """
    bs = s_batch.size(0)
    # Get interpolated points where grad penalty will be applied
    alpha = torch.rand(bs, 1, 1, 1).to(critic.device) # Interpolation parameter
    interpolates = (1-alpha)*s_batch + alpha*t_batch
    interpolates.requires_grad = True
    # Compute gradients
    inter_val = critic(interpolates)
    grads = autograd.grad(outputs=inter_val, 
                          inputs=interpolates,
                          grad_outputs=torch.ones_like(inter_val).to(critic.device),
                          create_graph=create_graph)[0]
    grads = grads.view(grads.size(0), -1)
    # Compute and return gradient penalty
    return (torch.clamp(grads.norm(2, dim=1)-1, min=0)**2).mean() * lamb    



def critic_trainer(critic, critters, source_loader, target_loader, propagator, loss_params):
    """
    Trains 'critic' (a network) for a number of iterations specified by 'critters' (an int). The critic is trained
    with a WGAN-GP type loss function (i.e. with a gradient penalty) to distinguish between samples obtained from the 
    'target_loader' and propagated samples obtained from the source loader (see below under Propagator what we mean by 
    'propagated'). The 'loss_params' argument should be a dict with the keys 'lr', 'betas', 'weight_decay' (for the Adam 
    optimizer) and 'lambda' (for the loss function). 
                                                                                                            
    Keeps track of the estimated Wasserstein distance and the gradient penalty values throughout training using stats_tracker.
    """    
    # Optimizer
    optimizer = optim.Adam(critic.parameters(), 
                           lr           = loss_params['lr'], 
                           betas        = loss_params['betas'], 
                           weight_decay = loss_params['weight_decay'])
    # Data iterators
    s_iter = iter(source_loader)
    t_iter = iter(target_loader)
    
    
    for i in tqdm(range(critters)):

        # Reset gradients (for safety only)
        for param in critic.parameters():
            param.grad = None # zero gradients of current critic
        
        ## DATA BATCHES
        s_batch, s_iter = get_data(s_iter, source_loader)
        s_batch = propagator.propagate_data(s_batch).to(critic.device) # Propagate source batch to current level
        t_batch, t_iter = get_data(t_iter, target_loader)
        t_batch = t_batch.to(critic.device)
        
        ## DIVERGENCE
        s_val = critic(s_batch).mean()
        t_val = critic(t_batch).mean()
        div_loss = t_val - s_val # Minimizing div_loss will maximize value on source batch and minimize on target batch
        div_loss.backward()
        
        ## GRADIENT PENALTY
        grad_pen = calc_grad_penalty(critic, s_batch, t_batch, loss_params['lambda'])
        grad_pen.backward()
        
        ## OPTIMIZE
        optimizer.step()
        
        # Reset gradients
        for param in critic.parameters():
            param.grad = None # zero gradients of current critic
        
        ## RECORD STATS
        div_loss *= -1
        wasser_d = div_loss + grad_pen
        stats_tracker.plot('training_wasser_D', wasser_d.detach().cpu())
        stats_tracker.plot('training_grad_penalty', grad_pen.detach().cpu())
        if i<critters-1:
            stats_tracker.tick() # For last critter, tick is done after recording some other stats, in train.py



def get_wasserstein_D(critic, num_batch, source_loader, target_loader, propagator, loss_params):
    """
    Computes and returns the Wasserstein distance between the target distribution and the propagated source
    distribution (see below under Propagator what we mean by 'propagated') as estimated by 'critic' over
    'num_batches' batches drawn from the source and target loader. The 'loss_params' argument is as in
    'critic_trainer' and is only used for its 'lambda' value.
    """
    # Data iterators
    s_iter = iter(source_loader)
    t_iter = iter(target_loader)
    wasser_d_sum = 0

    for _ in tqdm(range(num_batch)):
        ## DATA BATCHES
        s_batch, s_iter = get_data(s_iter, source_loader)
        s_batch = propagator.propagate_data(s_batch).to(critic.device) # Propagate sourec batch to current level
        t_batch, t_iter = get_data(t_iter, target_loader)
        t_batch = t_batch.to(critic.device)
        
        ## COMPUTE WASSERSTEIN_D
        s_val = critic(s_batch).mean()
        t_val = critic(t_batch).mean()
        grad_pen = calc_grad_penalty(critic, s_batch, t_batch, loss_params['lambda'], create_graph=False)
        wasser_d_sum += (s_val - t_val - grad_pen).detach()

    return (wasser_d_sum/num_batch).cpu().numpy()        



class Propagator:
    """
    Objects from this class (which we call 'propagators') are used to propagate data through pretrained TTC critics.
    A propagator has the attributes 'step_dict' and  'trained_critics'. The step_dict attribute contains a step_dict 
    as returned by 'unpack_step_info' function, which contains information on how to take TTC steps (e.g. step sizes
    and step types). For more information, see the unpack_step_info' function. The 'trained_critics' attribute is
    initialized as an empty list. New critics are meant to be added to this list once they are trained. 

    When calling any of the propagation functions on a batch of data, a propagator will take all the steps listed 
    in the 'step_dict' attribute for all the critics found in the 'trained_critics' attribute.      
    """
    def __init__(self, step_dict):
        self.step_dict = step_dict
        self.trained_critics = []
    
    def propagate_data(self, batch):
        """
        Only returns the batch after all steps on all trained critics have been taken.
        """
        for crit_idx, critic in enumerate(self.trained_critics):
            batch = batch.to(critic.device)
            for st, ty in zip(self.step_dict['step_size'][crit_idx],
                              self.step_dict['type'][crit_idx]):
                batch = step_master.take_step(batch, critic, st, ty)
        return batch
                
    def propagate_and_keep(self, batch):
        """
        Propagates batch and returns a list containing versions of the batch at all steps for which self.step_dict['keep_samples_step'][critic_index]
        is True. Ones needs to call add_bool_item with item_name='keep_samples_step' on the propagator's step_dict before calling
        this function.
        """
        returned_steps = []
        for crit_idx, critic in enumerate(self.trained_critics):
            batch = batch.to(critic.device)
            for st, ty, keep_step in zip(self.step_dict['step_size'][crit_idx],
                                         self.step_dict['type'][crit_idx],
                                         self.step_dict['keep_samples_step'][crit_idx]):
                batch = step_master.take_step(batch, critic, st, ty)
                if keep_step:
                    returned_steps.append(batch.detach().cpu())
        return returned_steps
        
    def propagate_and_save(self, save_folder, key, batch, batch_idx=None, batch_image=False, save_source=True, to_rgb=False, ext='pdf', grid_columns=16):
        """
        Propagates the batch and saves images at all steps for which self.step_dict['keep_samples_step'][critic_index] is True.
            - If batch_image=True, saves one image per specified step containing all elements of the batch arranged in a grid with 
              'grid_columns' columns. These images are save in the folder specified by 'save_folder'.
            - If batch_image=False, saves individual images of all elements in the batch at each specified step, under 
              f'{batch_idx * bs + i):06d}.{ext}' inside a sub-folder of 'save_folder' for the corresponding step.
            - If save_source=True, also saves an image of the input batch.
        """        
        if (batch_image==False) and (batch_idx is None):
            batch_idx=0

        if save_source:
            if batch_image:
                save_path = os.path.join(save_folder, f'source.{ext}')
                generate_samples.batch_image(save_path, batch.detach().cpu())
            else:
                source_folder = os.path.join(save_folder, 'source')
                os.makedirs(source_folder, exist_ok=True)
                generate_samples.save_individuals(source_folder, batch.detach().cpu(), batch_idx, to_rgb=to_rgb)
        
        step_num = 1
        for crit_idx, critic in enumerate(self.trained_critics):
            batch = batch.to(critic.device)
            for st, ty, save_step in zip(self.step_dict['step_size'][crit_idx],
                                                 self.step_dict['type'][crit_idx],
                                                 self.step_dict[key][crit_idx]):
                batch = step_master.take_step(batch, critic, st, ty)
                step_name = f'step_{step_num}_critic_{crit_idx+1}'
                step_num += 1
                if save_step:
                    # If creating one image for the whole batch, save that image under step name.
                    if batch_image:
                        save_path = os.path.join(save_folder, f'{step_name}.{ext}')
                        generate_samples.batch_image(save_path, batch.detach().cpu(), grid_columns)
                        continue
                    # If generating an individual image for each element of batch, save these images in a folder with the step name. 
                    step_folder = os.path.join(save_folder, f'{step_name}')
                    os.makedirs(step_folder, exist_ok=True)
                    generate_samples.save_individuals(step_folder, batch.detach().cpu(), batch_idx, to_rgb=to_rgb)
                    
            
        


# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
"""
Convenience functions for dealing with TTC step_dicts and the 'step_info' inputs in train.py and FID_eval.py.
"""

def check_num_critics(sp): # sp for 'step pattern'
    """
    Used by 'unpack_step_info'.
    """
    re_pattern = re.compile("^\([0-9]+\)$") # Regex pattern for specifying number of critics for a step pattern.
    if not bool(re_pattern.match(sp[0])): # First block of the step pattern doesn't specify a number of critics, so assume
        return 1, sp                      # number of critics is 1, keep step pattern as is.
    return int(sp[0][1:-1]), sp[1:] # Extract number of critics from first block and remove it from step pattern.
    


def unpack_step_block(sb, default_step_params): # sb for 'step block'
    """
    Used by 'unpack_step_info'.
    """
    re_pattern = re.compile("^[0-9]*\[[0-9a-z,.]+\]$")  # Regex pattern for valid step blocks that aren't just a number of steps.
    assert re_pattern.match(sb), f"Step block {sb} is invalid."
    
    reps = int(sb[:sb.find('[')]) if (sb.find('[') > 0) else 1 # Number of steps for this block.
    prefixes = ['ty', 'th'] # Correspond to step type and theta 
    sps = default_step_params.copy() # sps for 'step parameters'. Copy because we don't want to change the default step parameters.
    for i, param in enumerate(sb[sb.find('[')+1:-1].split(',')):
        if i > 1:
            raise ValueError(f"Step block {sb} is invalid: too many parameters given.")
        try:
            # Parameters specified as a int or float with no prefix is assumed to be in the correct position 
            # within the block parameters.
            sps[i] = float(param) 
        except ValueError:
            # Parameter specified with a prefix. 
            try:
                assert param[:2] in prefixes
                sps[prefixes.index(param[:2])] = float(param[2:])
            except ValueError:
                raise ValueError(f"The step parameter {param} in the step block {sb} is invalid.")
        
    # Sanity checks
    assert sps[0] in [1,2],   f"The type parameter in the step block {sb} is invalid. Should be 1 or 2."
    sps[0] = int(sps[0])        

    return [[param]*reps for param in sps]



def unpack_step_info(step_info, default_step_params, verbose=False):
    """
    This function is used to create the 'step_dict' dictionary that dictates how TTC steps are to be taken and
    that gets saved under step_dict.pkl.
    
    As described in the paper, it is sometimes useful to take several TTC steps on the same critic (refer to Algorithm 1
    in the paper for a clear explanation of the TTC algorithm). In addition to what is described in the paper, this codebase 
    implements two types of TTC steps (see step_master.py) and enables the use of the 'theta' parameter to modulate the step_sizes.
    The 'step_info' argument in train.py, which gets passed to the 'unpack_step_info' function here, allows one to conveniently 
    specify how many critics should be trained, how many TTC steps to take on each critic, as well as the step_type and theta
    parameter for each step. Note that all experiments reported in the paper used step_type=1 and theta=1.
    
        - The 'step_info' argument should be a string containing one or more "step_pattterns" separated (if there is more that one) by
          '__:__'. 
        - Each step pattern should start by an integer in parentheses, which specifies the number of critics that will follow this 
          step pattern. This should be followed by '_' and an arbitrary number of "step_blocks" separated by '_'.
        - Each step block should have the form n[ty,th] where:
                n is an integer specifying how many steps with these parameters to take,
                ty specifies the step type, it can be either 1 or 2,
                th specifies the theta parameter -- the step size will be theta * (estimated Wasserstein distance).
    
    EXAMPLES
        - step_info = '(20)_1[1,1]' : 
            20 critics on each of which one step of type 1 with theta=1 will be taken.
            Note that to obtain this, one can specify num_crit=20 and ignore the step_info argument when running train.py.
        - step_info = '(10)_2[1,1]__:__(20)_1[1,1]' :
            10 critics on which 2 steps will be taken followed by 20 critics on which 1 step will be taken (all steps
            are of type 1 with theta=1)
        - step_info = '(3)_2[1,1]_1[2,0.9]__:__(4)_5[1,0.7]' :
            2 critics on which will be taken 2 steps of type 1 with theta=1 followed by 1 step of type 2 with theta=0.9,
            then 4 critics on which will be taken 5 steps of type 1 with theta=0.7
    """
    step_dict = {'type':  [], 
                 'theta': []}
    param_keys = list(step_dict.keys()) # Only keys are 'type' and 'theta' for now. More keys to be added further down.

    for step_pattern in step_info.split('__:__'):
        step_pattern = step_pattern.split('_')
        num_crit, step_pattern = check_num_critics(step_pattern)
        block_params = [[] for _ in range(4)]
        for step_block in step_pattern:
            step_params = unpack_step_block(step_block, default_step_params)
            for i in range(2):
                block_params[i].extend(step_params[i])
        block_params = [[bp] * num_crit for bp in block_params]
        for i, key in enumerate(param_keys):
            step_dict[key].extend(block_params[i])
    num_crit = len(step_dict['type'])
    step_dict['step_size'] = [[] for _ in range(num_crit)] # Steps sizes to be computed and filled in during training
    step_dict['critic_params'] = []

    return step_dict



def add_bool_item(step_dict, where, item_name, reset=False):
    """
    Adds a list of lists of boolean values to 'step_dict' under 'item_names'. Each list in the list of lists
    corresponds to one critic, and each bollean value in that list corresponds to a step for that critic.
    
    This function is meant to be used in conjunction with the propagate_and_keep and propagate_and_save functions
    of the Propagator class. 
    
    The 'where' argument specifies which boolean values are to be set to True. 
        - If where='all_critics', the last element in each list is set to True, all others are set to False.
        - If where='last_step', only the last element in the last list is set to True, all others are set to False.
        - If where is anything else, all elements are set to True.
    
    If 'item_name' is already in step_dict.keys() when calling this function, then:
        - If reset=False, all elements of all pre-existing lists are set to False, regardless of whether they 
          were True or False before calling this function (this is useful when resuming from a previous experiment,
          e.g. to avoid recomputing FID at steps where is has already been computed).
        - If reset=True, the previous value of step_dict['item_name'] is deleted and add_bool_item acts as it would
          have if 'item_name' had not been in step_dict.
    """
    if (item_name not in step_dict) or reset:
        old_version = [[False]]
    else:
        old_version = step_dict[item_name]

    step_dict[item_name] = [[True for _ in step_dict['type'][i]] for i in range(len(step_dict['type']))]

    if where == 'all_critics':
        for step_list in step_dict[item_name]:
            for i in range(len(step_list)-1):
                step_list[i] = False
    if where == 'last_step':
        step_dict[item_name] = [[False for _ in step_dict['type'][i]] for i in range(len(step_dict['type']))]
        step_dict[item_name][-1][-1] = True 
 
    # Remove old
    for new_list, old_list in zip(step_dict[item_name], old_version):
        for i in range(len(old_list)):
            new_list[i] = new_list[i] and (not old_list[i])
    
    

def glue_step_dicts(first_dict, second_dict):
    """
    Used when resuming training from a previous experiment. 
    """
    step_dict = {}
    for key in first_dict.keys():
        if key in ['type', 'theta', 'step_size', 'critic_params']:
            step_dict[key] = first_dict[key] + second_dict[key]
        else:
            step_dict[key] = first_dict[key]
    return step_dict






# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

def check_folder(path, required_files):
    """
    Checks if 'path' already exists as a folder and, if so, what's in it. If 'path' does not already exists,
    it gets created as an empty folder.
    BEHAVIOR:
        - If 'path' does not already exist: Creates 'path' as an empty folder and then returns False.
        - If 'path' already exists but not as a folder: Throws OSError.
        - If 'path' already exists as a folder which is empty: Returns False.
        - If 'path' already exists as a folder which is not empty but does not contain all required files:
          Throws OSError.
        - If 'path' already exists as a folder which contains all files in 'required_files': Returns True.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return False 
    
    if not os.path.isdir(path):
        raise OSError("Given folder path already exists but is not a directory.")
    
    if not os.listdir(path): # Path is empty
        return False 
    
    if not all(file in os.listdir(path) for file in required_files):
        raise OSError("Given folder already exists, but does not contain all necessary files and is not empty.")
    
    return True 