import os
import torch
from torchvision import transforms, datasets


def dataloader_tool(data, bs, shape=[None,None,None], pretransforms=None, device=None, train=True, random_crop=False, num_workers=0):
    """
    Returns a dataloader based on the data argument, which should be either 'noise' or a path leading to a dataset.
        - If data == 'noise', returns a noise loader. The shape argument must be specified.
        - If data is a path, returns a dataloader based on what is found. If MNIST/... is there, returns MNIST loader.
          Else if FashionMNIST/... is there, returns a FashionMNIST loader. Else, returns a generic ImageFolder loader.
          If the shape argument is unspecified, the shape is deduced from the images in the folder (this assumes all images are the same size).
          If the shape argument is specified, will resize the arguments with torchvision.transforms.Resize if random_crop == False, 
          else with torchvision.transforms.RandomCrop.
    """    
    assert shape[0] in [1, 3, -1, None], "Invalid number of color channels given - must be 1, 3 or -1 or None for inferring from data."
    shape = [None if s==-1 else s for s in shape] # Replace -1's by None's
    
    if data=='noise':
        assert all(type(s)==int for s in shape) and (len(shape)==3), \
            'For noise loader, shape argument must be specified (a list of 3 integers: [num_chan, height, width]).'
        return NoiseLoader(bs, shape, device)
    
    # Else, data must be path of folder containing dataset
    assert os.path.isdir(data), "data argument should be 'noise' or the path of a folder containing the data"
    
    # Is MNIST here?
    if 'MNIST' in os.listdir(data):            
        return mnist(data, bs, shape, pretransforms, device, train, num_workers)
    
    # Is Fashion-MNIST here?
    if 'FashionMNIST' in os.listdir(data): 
        return fashion(data, bs, shape, pretransforms, device, train, num_workers)
    
    # Otherwise, assume ImageFolder format
    else:
        return image_folder(data, bs, shape, pretransforms, device, train, random_crop, num_workers)
        

class NoiseLoader:
    def __init__(self, bs, shape, device):
        self.bs = bs
        self.shape = shape
        self.dataset_length = float('inf') # Ain't no end to the noise
    
    def __iter__(self):
        return self

    def __next__(self):
        return torch.randn([self.bs, self.shape[0], self.shape[1], self.shape[2]]), None # second is dummy value


def mnist(data, bs, shape, pretransforms, device, train, num_workers=0):
    # Preprocessing
    transforms_ = []
    height = 28 if (shape[1] is None) else shape[1]
    width  = 28 if (shape[2] is None) else shape[2]
    if (height, width) != (28, 28):
        transforms_.append(transforms.Resize((height, width)))
    transforms_.append(transforms.ToTensor())
    if shape[0]==3:
        shape = [3, height, width]
        transforms_.append(transforms.Lambda(lambda x: torch.cat([x,x,x], dim=0)))
        transforms_.append(transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)))
    else:
        shape = [1, height, width]
        transforms_.append(transforms.Normalize((0.5),(0.5)))    
    if pretransforms is not None:
        for pt in pretransforms:
            transforms_.append(pt)
    preprocess = transforms.Compose(transforms_)
    
    # Dataloader
    ds = datasets.MNIST(data, download=False, train=train, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size = bs,
                                             drop_last = True,
                                             shuffle = True,
                                             num_workers = num_workers,
                                             pin_memory = True)
    dataloader.bs = bs
    dataloader.shape = shape
    dataloader.type = 'mnist'
    dataloader.dataset_length = len(ds)
    return dataloader



def fashion(data, bs, shape, pretransforms, device, train, num_workers=0):
    # Preprocessing
    transforms_ = []
    height = 28 if (shape[1] is None) else shape[1]
    width  = 28 if (shape[2] is None) else shape[2]
    if (height, width) != (28, 28):
        transforms_.append(transforms.Resize((height, width)))
    transforms_.append(transforms.ToTensor())
    if shape[0]==3:
        shape = [3, height, width]
        transforms_.append(transforms.Lambda(lambda x: torch.cat([x,x,x], dim=0)))
        transforms_.append(transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)))
    else:
        shape = [1, height, width]
        transforms_.append(transforms.Normalize((0.5),(0.5)))    
    if pretransforms is not None:
        for pt in pretransforms:
            transforms_.append(pt)
    preprocess = transforms.Compose(transforms_)
    
    # Dataloader
    ds = datasets.FashionMNIST(data, download=False, train=train, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size = bs,
                                             drop_last = True,
                                             shuffle = True,
                                             num_workers = num_workers,
                                             pin_memory = True)
    dataloader.bs = bs
    dataloader.shape = shape
    dataloader.type = 'fashion'
    dataloader.dataset_length = len(ds)
    return dataloader

    

def image_folder(data, bs, shape, pretransforms, device, train, random_crop=False, num_workers=0):      
    if ('train' in os.listdir(data)) and ('test' in os.listdir(data)):
        if train or (train is None):
            data = os.path.join(data, 'train')
        else:
            data = os.path.join(data, 'test')
    
    # Check shape of data (Assumes all images have save shape)
    to_tens = transforms.ToTensor()
    ds = datasets.ImageFolder(data)
    data_shape = to_tens(ds[0][0]).shape
    
    # Preprocessing
    transforms_ = []
    height = data_shape[1] if (shape[1] is None) else shape[1]
    width  = data_shape[2] if (shape[2] is None) else shape[2]
    if [height, width] != data_shape[1:]:
        if random_crop:
            transforms_.append(transforms.RandomCrop((height, width)))
        else:
            transforms_.append(transforms.Resize((height, width)))
    transforms_.append(to_tens)
    if (shape[0] is not None) and (data_shape[0] > shape[0]): # Data is RGB but want grayscale
        shape = [1, height, width]
        transforms_.append(transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True)))
        transforms_.append(transforms.Normalize((0.5),(0.5)))
    elif (shape[0] is not None) and (data_shape[0] < shape[0]): # Data is grayscale but want RGB
        transforms_.append(transforms.Lambda(lambda x: torch.cat([x,x,x], dim=0)))
        transforms_.append(transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)))
    else:
        shape = [data_shape[0], height, width]
        norma = tuple(0.5 for _ in range(data_shape[0]))
        transforms_.append(transforms.Normalize(norma,norma))
    if pretransforms is not None:
        for pt in pretransforms:
            transforms_.append(pt)
    preprocess = transforms.Compose(transforms_)
    
    # Dataloader
    ds = datasets.ImageFolder(data, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size = bs,
                                             drop_last = True,
                                             shuffle = True,
                                             num_workers = num_workers,
                                             pin_memory = True)
    dataloader.bs = bs
    dataloader.shape = shape
    dataloader.type = 'image_folder'
    dataloader.dataset_length = len(ds)
    return dataloader