# Python
import sys
sys.path.extend(['../../'])
import warnings
from gpytorch.utils.errors import NanError
from gpytorch.utils.warnings import NumericalWarning

# custom 
import config as cg

# Torch
import torch
import torch.distributions as td

## ================================================= ##
##  Log Batched Multivariate Gaussian: log N(x|mu,C) ##
def batched_log_Gaussian(obs: torch.tensor, mean: torch.tensor, cov: torch.tensor, diagonal:bool, cov_is_inverse: bool)-> torch.tensor:
    """
    Computes a batched of * log p(obs|mean,cov) where p(y|f) is a  Gaussian distribution, with dimensionality N. 
    Returns a vector of shape *.
    -0.5*N log 2pi -0.5*\log|Cov| -0.5[ obs^T Cov^{-1} obs -2 obs^TCov^{-1} mean + mean^TCov^{-1}mean]
            Args: 
                    obs            :->: random variable with shape (*,N)
                    mean           :->: mean -> matrix of shape (*,N)
                    cov            :->: covariance -> Matrix of shape (*,N) if diagonal=True else batch of matrix (*,N,N)
                    diagonal       :->: if covariance is diagonal or not 
                    cov_is_inverse :->: if the covariance provided is already the inverse
    
    #TODO: Check argument shapes
    """

    N = mean.size(-1)
    cte =  N*torch.log(2*cg.pi.to(cg.device).type(cg.dtype))
    
    if diagonal:

        log_det_C = torch.sum(torch.log(cov),-1)
        inv_C = cov
        if not cov_is_inverse:
            inv_C = 1./cov # Inversion given a diagonal matrix. Use torch.cholesky_solve for full matrix.
        else:
            log_det_C *= -1 # switch sign

        exp_arg = (obs*inv_C*obs).sum(-1) -2 * (obs*inv_C*mean).sum(-1) + (mean*inv_C*mean).sum(-1)

    else:
        raise NotImplemented("log_Gaussian for full covariance matrix is not implemented yet.")
    return -0.5*( cte + log_det_C + exp_arg )


## =============================================
## Add Jitter for safe cholesky decomposition ##
def add_jitter_MultivariateNormal(mu_q_f,K):
    # -> Copied from gpytorch: https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/utils/cholesky.py
    #    We have to add it sometimes because td.MultivariateNormal does not add jitter when necessary

    # TODO: give support for diagonal instance of Multivariate Normals. 

    jitter = 1e-6 if K.dtype == torch.float32 else 1e-8
    Kprime = K.clone()
    jitter_prev = 0
    for i in range(5):
        jitter_new = jitter * (10 ** i)
        Kprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
        jitter_prev = jitter_new
        try:
            q_f = td.multivariate_normal.MultivariateNormal(mu_q_f, Kprime)
            return q_f
        except RuntimeError: 
            continue
    raise RuntimeError("Cannot compute a stable td.MultivariateNormal instance. Got singular covariance matrix")

## ========================================================================= ##
## Safe cholesky from gpytorch but returning the covariance with jitter also ##
def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        if cg.constant_jitter is not None:
            A.diagonal(dim1=-2, dim2=-1).add_(cg.constant_jitter)

        L = torch.cholesky(A, upper=upper, out=out)

        ## For a weird reason sometimes even A has nan torch.cholesky doesnt fail and goes into except. We check it manually and raise it ourselves
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        return L, A
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(3):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(f"A not p.d., added jitter of {jitter_new} to the diagonal", NumericalWarning)
                return L, Aprime
            except RuntimeError:
                continue
        raise e
