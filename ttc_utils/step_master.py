"""
Functions governing how TTC steps are taken.
"""

import torch


def grad_calc(batch, critic):
    """Returns the gradients of critic at batch"""
    batch = batch.detach().clone()
    batch.requires_grad = True
    vals = critic(batch)
    grads = torch.autograd.grad(outputs = vals, 
                                inputs = batch,
                                grad_outputs = torch.ones(vals.size()).to(critic.device))[0]
    return grads



def take_step(batch, critic, step_size, type_):
    """
    Returns the batch obtained bby taking a TTC step of size 'step_size' on 'critic' starting from 'batch'.
    We only use type_=='1' in the paper.
    """
    step_type = {'1':'ttc_step', '2':'normalized_grads_step'}
    return globals()[step_type[str(type_)]](batch, critic, step_size)



def ttc_step(batch, critic, step_size):
    """
    Main step type used for all TTC experiments reported in the paper. Corresponds to step_type='1'.
    """
    grads = grad_calc(batch, critic)
    batch = batch - step_size*grads
    return batch.detach()



def normalized_grads_step(batch, critic, step_size, tolerance):
    """
    Steps with normalized gradients. Corresponds to step_type='2'.
    """
    bs = batch.shape[0]
    grads = grad_calc(batch, critic).view(bs,-1)
    grads = grads/torch.norm(grads, dim=1, keepdim=True)
    grads = grads.view(batch.shape)
    batch = batch - step_size*grads
    return batch