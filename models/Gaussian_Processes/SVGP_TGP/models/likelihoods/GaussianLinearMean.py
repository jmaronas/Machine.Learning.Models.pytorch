#-*- coding: utf-8 -*-
# GaussianLinearMean.py : Keeps a Linear Gaussian Mean p(y|f) likelihood with homoceodastic noise.
# Author: Juan Maroñas 

# Python
import sys
sys.path.extend(['../../../../'])

# Torch
import torch
import torch.nn as nn
import torch.distributions as td

# custom
import config as cg
from ..utils import batched_log_Gaussian


class GaussianLinearMean(nn.Module):
    ''' Place a GP over the mean of a Gaussian likelihood $p(y|f)$ 
        with noise variance $\sigma^2$ and with a linear transformation 
        (identity) over $f$. It supports multi-output (independent) 
        GPs and the possibility of sharing the noise between the different outputs. 
    '''

    def __init__(self,out_dim : int, noise_init: float, noise_is_shared : bool):
        super(GaussianLinearMean,self).__init__()

        self.out_dim = out_dim
        self.noise_is_shared = noise_is_shared

        if noise_is_shared: # if noise is shared create one parameter and expand to out_dim shape
            log_var_noise = nn.Parameter(torch.ones(1,1,dtype = cg.dtype)*torch.log(torch.tensor(noise_init,dtype = cg.dtype)))

        else: # creates a vector of noise variance parameters.
            log_var_noise = nn.Parameter(torch.ones(out_dim,1,dtype = cg.dtype)*torch.log(torch.tensor(noise_init,dtype = cg.dtype)))

        self.log_var_noise = log_var_noise

    def expected_log_prob(self, Y, gauss_mean, gauss_cov, **kwargs):
        """ Expected Log Likelihood

            Computes E_q(f) [\log p(y|f)] = \int q(f) \log p(y|f) df = \log N(y|\mu_f,C_y) -0.5 TR[C_y^{-1}C_f]

                - C_y represents the covariance of the observation model. 
                - C_f represents the covaraince from the variational distribution
                -Acts on batched form. Hence returns vector of shape (Dy,)

            Args:

                `Y`          (torch.tensor)  :->:  Labels representing the mean. Shape (Dy,MB)
                `gauss_mean` (torch.tensor)  :->:  mean from q(f). Shape (Dy,MB)
                `gauss_cov`  (torch.tensor)  :->:  diagonal covariance from q(f). Shape (Dy,MB)

        """
        if self.noise_is_shared:
            log_var_noise = self.log_var_noise.expand(self.out_dim,1)
        else:
            log_var_noise = self.log_var_noise

        N = Y.size(1)
        C_y_inv = (1./torch.exp(log_var_noise)).expand(-1,N)

        log_p_y = batched_log_Gaussian( obs = Y, mean = gauss_mean, cov = C_y_inv, diagonal = True, cov_is_inverse = True)
        trace     =  -0.5 * torch.sum(torch.mul(C_y_inv,gauss_cov),1) # The trace of product of two diagonal matrix is the sum of the element-wise product
        ELL = log_p_y + trace
        return ELL

    def marginal_moments(self,gauss_mean,gauss_cov, diagonal, **kwargs):
        """ Computes the moments of order 1 and non centered 2 of the observation model integrated out w.r.t a Gaussian with means and covariances. 

            p(y|x) = \int p(y|f) p(f) df

            - Note that p(f) can be any distribution, hence this is useful also for the marginal likelihood

            Args:
                `gauss_mean` (torch.tensor)  :->: mean from q(f). Shape (Dy,MB)
                `gauss_cov`  (torch.tensor)  :->: covariance from q(f). Shape (Dy,MB) if diagonal is True, else (Dy,MB,MB)
                `diagonal`   (bool)          :->: wether the covariance from the distribution is diagonal or not

        """
        N = gauss_mean.size(1)

        if self.noise_is_shared:
            log_var_noise = self.log_var_noise.expand(self.out_dim,1)
        else:
            log_var_noise = self.log_var_noise

        C_Y = torch.exp(log_var_noise).expand(-1,N) # shape (Dy,MB)

        if not diagonal:
            C_Y = torch.diag_embed(C_Y) # shape (Dy,MB,MB)

        C_Y  = C_Y + gauss_cov
        mu_Y = gauss_mean.clone() 


        return mu_Y, C_Y


