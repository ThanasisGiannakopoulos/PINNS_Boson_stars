# utils.py
import torch # type: ignore
import numpy as np # type: ignore
import os

def create_directory(path):
    os.makedirs(path, exist_ok=True)
    print(f"Directory created or already exists: {path}")

def random_domain_points(n):
    xhigh = 0.5 * torch.rand((int(n / 2), 1), requires_grad=True) + 0.5  # [0.5,1)
    xlow = -0.5 * torch.rand((int(n / 2), 1), requires_grad=True) + 0.5  # (0,0.5]
    x = torch.cat((xlow, xhigh), 0)
    return x

def gradients(outputs, inputs, order=1):
    if order == 1:
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    elif order > 1:
        return gradients(gradients(outputs, inputs, 1), inputs, order - 1)
    else:
        return outputs
