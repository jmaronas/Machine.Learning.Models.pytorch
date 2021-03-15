#-*- coding: utf-8 -*-
# SVGP.py : file containing base class for the sparse GP model as in Hensman et al.
# Author: Juan Maroñas

# Python
import sys
sys.path.extend(['../'])

# Torch
import torch
import torch.nn as nn

# Module specific
from .TGP  import TGP
from .flow import instance_flow

## Sparse GP Hensman et al
class SVGP(TGP):
    def __init__(self,model_specs: list, init_Z: torch.tensor, N: float, likelihood : nn.Module, num_outputs: int, is_whiten: bool, Z_is_shared : bool ) -> None:
        """
        Stochastic Variational Sparse GP Hensman et al
                Args: 
                        :attr:  
                                `model_specs`         (list)         :->: tuple (A,B) where A and B are string representing the desired mean and covariance functions.
                                                                          For the moment all the GPs at a layer shared the functional form of these
                                `init_Z`              (torch.tensor) :->: initial inducing point locations
                                `N`                   (float)        :->: total training size	
                                `likelihood`          (nn.Module)    :->: Likelihood instance that will depend on the task to carry out
                                `num_outputs`         (int)          :->: number of output GP. The number of inputs is taken from the dimensionality of init_Z
                                `is_whiten`           (bool)         :->: use whitened representation of inducing points.
                                `Z_is_shared`         (bool)         :->: True if the inducing point locations are shared


        """
        flow_specs      = [ instance_flow([('identity',[])])  for i in range(num_outputs) ]
        super(SVGP, self).__init__( model_specs, init_Z, N, likelihood, num_outputs, is_whiten, Z_is_shared, flow_specs, be_fully_bayesian = False ) 

