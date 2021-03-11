#-*- coding: utf-8 -*-
# config.py : Config file to be used by all the code
# Author: Juan Maroñas 

import torch
import numpy
import math
import platform
import pkg_resources

def check_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device

def check_torch(torch_version):
    pkg_resources.require("torch>={}".format(torch_version))
    #if torch.__version__ != torch_version:
    #    raise ImportError('Torch does not match correct version {}'.format(torch_version))

def reset_seed(seed: int) -> None: 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)

## Config Variables
torch_version = '1.7.0'
device=check_device()
dtype = torch.float32
torch.set_default_dtype(dtype)
is_linux = 'linux' in platform.platform().lower()
reset_seed(seed=1)

## Constant definitions
pi = torch.tensor(math.pi).to(device)

## Callers
check_torch(torch_version)


