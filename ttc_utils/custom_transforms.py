import torch


# Adds mean zero noise with variance sigma^2 to tensor
class AddGaussianNoise:
    
    def __init__(self, std=1.):
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std
    
    def __repr__(self):
        return self.__class__.__name__ + f'(std = {self.std})'



