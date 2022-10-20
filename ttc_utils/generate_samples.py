import torch
from torchvision import utils
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')



def save_individuals(save_folder, batch, batch_idx, to_rgb=False, ext='jpg'):
    """
    Will save the individual images contained in 'batch' inside the folder specified by 'save_folder', under 
    f'{batch_idx * bs + i):06d}.{ext}'. 
    """
    batch = 0.5*batch + 0.5*torch.ones_like(batch) # by default, data is generated in [-1,1]
    bs = batch.shape[0]
    if (to_rgb and (batch.shape[1]==1)):
        batch = batch.repeat(1,3,1,1)
    for i in range(bs):
        utils.save_image(batch[i], os.path.join(save_folder, f'{(batch_idx * bs + i):06d}.{ext}'))



def batch_comparisons(save_folder, batches, grid_columns=0, ext='pdf', dpi=300):
    """
    Will save individual images placing the corresponding elements in each batch in 'batches' side by side. This is useful 
    to create images like those shown in the paper for denoising, translation and deblurring. If 'grid_columns' > 0,
    will also save a grid with all the individual images arranged in a grid with 'grid_columns' columns.
        - 'batches' should be a list of batches of images, each containing the same number of images of the same size.
        - 'save_folder' is the path of the folder in which all images will be saved.
    """
    buffer_width = 2
    batches = [0.5*batch + 0.5*torch.ones_like(batch) for batch in batches]
    num_chan = batches[0].shape[1]
    height = batches[0].shape[2]
    buffer = 0.2 * torch.ones((num_chan, height, buffer_width))
    
    if grid_columns > 0: # Then, also save a grid with all the saved images
        grid_batch = None
    
    for i, ims in enumerate(zip(*batches)):
        # Glue images together, with buffer between them
        saved_im = ims[0]
        for im in ims[1:]:
            saved_im = torch.cat((saved_im, buffer), dim=2)
            saved_im = torch.cat((saved_im, im), dim=2)

        # Save image
        utils.save_image(saved_im, os.path.join(save_folder, f'{i}.{ext}'))
        
        if grid_columns > 0: # Then, also save a grid with all the saved images
            if i==0:
                grid_batch = saved_im.view((1,*saved_im.shape))
            else:
                grid_batch = torch.cat((grid_batch, saved_im.view((1,*saved_im.shape))), dim=0)
    
    if grid_columns > 0:
        grid = utils.make_grid(grid_batch, nrow=grid_columns, padding=4)
        plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.savefig(os.path.join(save_folder, f'all_images.{ext}'), dpi=dpi)
        plt.close()
        
                

def batch_image(save_path, batch, grid_columns=16):
    """
    Will save all the images contained in 'batch' in a single grid with 'grid_columns' columns, under a path specified by
    'save_path'. Note that 'save_path' should not be a folder, but the full path under which the grid will be saved (unlike
    the 'save_folder' argument of 'save_individuals' and 'batch_comparison').
    """
    batch = 0.5*batch + 0.5*torch.ones_like(batch) # by default, data is generated in [-1,1]
    grid = utils.make_grid(batch, nrow=grid_columns, padding=1)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.savefig(save_path, dpi=300)
    plt.close()