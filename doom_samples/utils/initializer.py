import torch
from torch import nn
import numpy as np

def init_weights(module):
    if isinstance(module, nn.Linear):
        linear_init = prepare_linear_init(0.01)
        module.weight.data = linear_init(module.weight.shape)
        module.bias.data = torch.zeros(module.bias.shape, requires_grad=True)
    if isinstance(module, nn.Conv2d):
        module.weight.data = conv_init(module.kernel_size, module.in_channels, module.out_channels, module.weight.shape)
        module.weight.bias = torch.zeros(module.bias.shape, requires_grad=True)

def prepare_linear_init(std):
    def linear_init(shape_of_weights):
        out = torch.randn(shape_of_weights).to(torch.float32).numpy()
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return torch.tensor(out, requires_grad=True)
    return linear_init

def conv_init(kernel_size, in_fmap_quantity, out_fmap_quantity, shape_of_weights):
    fan_in = kernel_size[0]*kernel_size[1]*in_fmap_quantity
    fan_out = kernel_size[0]*kernel_size[1]*out_fmap_quantity
    w_bound = torch.tensor(np.sqrt(6./(fan_in+fan_out)))
    uniformilly_distributed_weights = 2*(torch.rand(shape_of_weights, requires_grad=True)-1/2)*w_bound
    return uniformilly_distributed_weights