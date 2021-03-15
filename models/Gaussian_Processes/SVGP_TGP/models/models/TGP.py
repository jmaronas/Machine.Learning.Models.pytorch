#-*- coding: utf-8 -*-
# TGP.py : file containing base class for the TGP model
# Author: Juan Maroñas

def enable_eval_dropout(modules):
    is_dropout = False
    for module in modules:
        if 'Dropout' in type(module).__name__:
            module.train()
            is_dropout = True
    return is_dropout


# Python
import sys
sys.path.extend(['../'])
sys.path.extend(['../../../../'])

# Standard
import numpy

# Torch
import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions.kl import kl_divergence

# Gpytorch
import gpytorch  as gpy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.utils.broadcasting import _pad_with_singletons
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D

# Pytorch Library
from pytorchlib import compute_calibration_measures

# Custom repository Machine Learning
import config as cg

# Custom this module
from ..utils import batched_log_Gaussian, add_jitter_MultivariateNormal, psd_safe_cholesky
from ..likelihoods import GaussianLinearMean, GaussianNonLinearMean, MulticlassCategorical, Bernoulli

# Module specific
from .flow import instance_flow


## Sparse TGP. Warping prior p(f) using normalizing flows
class TGP(nn.Module):
    def __init__(self,model_specs: list, init_Z: torch.tensor, N: float, likelihood : nn.Module, num_outputs: int, is_whiten: bool, Z_is_shared: bool, flow_specs: list,  be_fully_bayesian : bool ) -> None:
        """
                Args: 
                        :attr:  `model_specs`         (list)         :->: tuple (A,B) where A is a string representing the mean used and B is a kernel instance.
                                                                          This kernel instance should have batch_shape = number of outputs if K_is_shared = False,
                                                                          and batch_shape = 1 else. For the moment all the GPs at a layer shared the functional form of these.
                                `X`                   (torch.tensor) :->: Full training set (or subset) of samples used for the SVD 
                                `init_Z`              (torch.tensor) :->: initial inducing point locations
                                `N`                   (float)        :->: total training size	
                                `likelihood`          (nn.Module)    :->: Likelihood instance that will depend on the task to carry out
                                `num_outputs`         (int)          :->: number of output GP. The number of inputs is taken from the dimensionality of init_Z
                                `is_whiten`           (bool)         :->: use whitened representation of inducing points.
                                `Z_is_shared`         (bool)         :->: True if the inducing point locations are shared
                                `flow_specs`          (list)         :->: A list of list containing lists of strings (or flows instances) specifying the composition 
                                                                          and interconnection of flows per output dimension. The list contains num_output lists, specifying the flows per output GP.
                                `be_fully_bayesian`   (bool)         :->: If true, then input dependent flows are integrated with monte carlo dropout when possible


          # -> Some notes for deciding if whitenning or not: https://gpytorch.readthedocs.io/en/latest/variational.html#gpytorch.variational.UnwhitenedVariationalStrategy	
        """
        super(TGP, self).__init__()
        ## ==== Check assertions ==== ##
        assert len(model_specs) == 2, 'Parameter model_specs should be len 2. First position string with the mean and second position string with the kernels'

        ## ==== Config Variables ==== ##
        self.out_dim          = int(num_outputs)    # output dimension
        self.inp_dim          = int(init_Z.size(1)) # input dimension
        self.Z_is_shared      = Z_is_shared         # if the inducing points are shared 
        self.N                = float(N)            # training size
        self.M                = init_Z.size(0)      # number of inducing points
        self.likelihood       = likelihood
        
        self.fully_bayesian   = be_fully_bayesian

        ## ==== Tools ==== ##
        self.standard_sampler = td.MultivariateNormal(torch.zeros(1,).to(cg.device),torch.eye(1).to(cg.device))  # used in the reparameterization trick.

        if isinstance(self.likelihood, GaussianNonLinearMean):
            self.quad_points = self.likelihood.quad_points 
            self.quad        = GaussHermiteQuadrature1D(self.quad_points) # quadrature integrator. 
        else:
            self.quad_points = cg.quad_points
            self.quad        = GaussHermiteQuadrature1D(self.quad_points)
            

        ## ==== Set the Model ==== ##
        # Variational distribution
        self.initialize_inducing(init_Z)	
        self.initialize_variational_distribution(is_whiten)

        # Model distribution
        self.mean_function       = model_specs[0]
        self.covariance_function = model_specs[1]

        G_matrix                 =  self.initialize_flows(flow_specs)
        self.G_matrix            = G_matrix

    ## ================================ ##
    ## == Some configuration Methods == ## 
    ## ================================ ##

    ## ========================================== ##
    ##  Set model to work in fully bayesian mode
    def be_fully_bayesian(self, mode):
        self.fully_bayesian = mode

    ## ================================================ ##
    ##   Variational distribution q(f,u) = p(f|u)q(u)   ##

    ### Inducing point locations Z ###
    def initialize_inducing(self,init_Z: torch.tensor) -> None:
        """Initializes inducing points using the argument init_Z. It prepares this inducing points
           to be or not to be shared by all kernels.
        """
        if self.Z_is_shared:
            self.Z = nn.Parameter(init_Z.unsqueeze(dim = 0)) # this parameter is repeated when needed
        else:
            Z = torch.zeros(self.out_dim,self.M,self.inp_dim)
            for l in range(self.out_dim):
                aux_init = init_Z.clone()
                Z[l,:] = aux_init
            self.Z = nn.Parameter(Z)

    ## Variational Distribution q(U)
    def initialize_variational_distribution(self,is_whiten:bool) -> None:
        """Initializes the GP variational distribution for work with batched inputs and not be or not to be shared.
           It initializes its parameters using the init_dict passed as argument or the default values at config.models
        """

        o_d = self.out_dim
        q_U =  CholeskyVariationalDistribution(self.M,batch_shape = torch.Size([o_d]))

        init_S_U = torch.eye(self.M, self.M).view(1,self.M,self.M).repeat(o_d,1,1)*numpy.sqrt(1e-5) # take sqrt as we parameterize the lower triangular
        init_m_U = torch.ones(o_d,self.M)*0.0

        q_U.chol_variational_covar.data = init_S_U
        q_U.variational_mean.data = init_m_U

        self.is_whiten = is_whiten
        self.q_U = q_U


    ## ========================== ##
    ## === Model Distribution === ##

    ## == Flows == ##
    ## Initialize Flow. G (prior) in the paper ##
    def initialize_flows(self, flow_specs) -> None:
        """ Initializes the flows applied on the prior . Flow_specs is a list with an instance of the flow used per output GP.
        
        """

        G_matrix = []

        for idx,fl in enumerate(flow_specs):
            G_matrix.append(fl)

        G_matrix        = nn.ModuleList(G_matrix)

        return G_matrix

    ## ============================ ##
    ## ==  FINISH CONFIG METHODS == ##
    ## ============================ ##

    ## =============================== ##
    ## == MODEL COMPUTATION METHODS == ## 
    ## =============================== ##

    def marginal_variational_qf_parameters(self, X : torch.tensor, diagonal : bool, is_duvenaud: bool, init_Z : torch.tensor = None) -> torch.tensor:
        """ Marginal Variational posterior q(f) = \int p(f|u) q(u) d_u
            q(f) = int p(f|u) q(u) d_u = N(f|K_xz K_zz_inv m + m_x -K_xz K_zz_inv \mu_z, 
                                                 K_xx -K_xz K_zz_inv K_zx + [K_xz K_zz_inv] S [K_xz K_zz_inv]^T)
                Args:
                        `X`           (torch.tensor)  :->:  input locations where the marginal distribution q(f) is computed. Can hace shape (S*MB,Dx) or (Dy,S*MB,Dx)
                        `diagonal`    (bool)          :->:  If true, return only the diagonal covariance
                        `is_duvenaud` (bool)          :->:  Indicate if we are using duvenaud mean function. Only useful in DGPs
                        `init_Z`      (torch.tensor)  :->:  Only used if is_duvenaud = True. It is used to concatenate the input inducing points to 
                                                            the inducing points at each layer

                Returns:
                        `mu_q_f`      (torch.tensor)  :->:  shape Dy,MB,1
                        `cov_q_f`     (torch.tensor)  :->:  shape (Dy,MB,1) if diagonal else (Dy,MB,MB)
        """
        ## ================================= ##
        ## ===== Pre-Compute Variables ===== ##
        if len(X.shape) == 2:
            X = X.repeat(self.out_dim,1,1) # repeat here as if not this operation will be done twice by the marginal_qf_parameter and likelihood to work batched and multioutput respectively
        assert len(X.shape) == 3, 'Invalid input X.shape' 

        Dy,MB,M  = self.out_dim,X.size(1),self.M
        Z        = self.Z

        kernel   = self.covariance_function
        mean     = self.mean_function

        if self.Z_is_shared:
            # In this case this repeat is not particulary needed because the kernel will repeat Z
            # when doing forward both if batch_shape is out_dim or is 1 (self.kernel_is_shared True)
            # Keep it explicitely for better understanding of the code.
            Z = Z.repeat(self.out_dim,1,1) 

        # Concatenate inducing points if is duvenaud
        if is_duvenaud:
            #z_concat = X[0,0:self.M,-1].view(self.M,1)
            init_Z    = init_Z.view(1,self.M,-1).repeat(self.out_dim,1,1)
            Z = torch.cat((Z,init_Z),2)

        K_xx = kernel(X,are_equal = True, diag = diagonal)
        mu_x = gpy.lazy.delazify( mean(X) ).view(Dy, MB, 1)

        K_zz = kernel(Z,are_equal = False).evaluate()
        mu_z = gpy.lazy.delazify( mean(Z) ).view(Dy, M , 1)

        K_xz = kernel(X,Z,are_equal = False).evaluate()

        # stabilize K_xz. In case Z = X we should add jitter if psd_safe_cholesky adds jitter to K_zz
        # jitter can only be added to square matrices

        K_zx = torch.transpose(K_xz,1,2) # pre-compute the transpose as it is required several times

        # cholesky from K_zz
        L_zz, K_zz  = psd_safe_cholesky(K_zz, upper = False, jitter = cg.global_jitter) # The K_zz returned is that with noise

        if self.is_whiten:
            L_zz_t = L_zz.transpose(1,2) 

        # variational distribution
        q_U    = self.q_U
        m_q_U  = q_U.variational_mean
        K_q_U  = q_U.chol_variational_covar
        
        lower_mask = torch.ones(K_q_U.shape[-2:], dtype=cg.dtype, device=cg.device).tril(0)
        L_q_U = K_q_U.mul(lower_mask)
        K_q_U = torch.matmul( L_q_U,L_q_U.transpose(1,2) )
        m_q_U  = m_q_U.view(Dy,M,-1)

        ## =================== ##
        ## ==== mean q(f) ==== ##

        if self.is_whiten:
            # mu_qf = K_{xz}[L_{zz}^T]^{-1}m_0+\mu_x
            sol,_ = torch.triangular_solve(m_q_U, L_zz_t, upper = True)
            mu_q_f = torch.matmul(K_xz,sol) + mu_x

        else:
            # mu_qf = K_xz K_zz_inv( m - mu_Z) + m_x
            lhs = torch.cholesky_solve(m_q_U-mu_z, L_zz, upper = False)
            mu_q_f = torch.matmul(K_xz,lhs) + mu_x

        
        ## ========================= ##
        ## ==== covariance q(f) ==== ##
        ## Note:
            # To compute the diagonal q(f) we perform the following identity. Here @ indicates matrix product and .* element-wise product
            # For K_xz @ K_zz_inv @ K_zx the diagonal is:
            #   sum(K_zx .* [K_zz_inv @ K_zx],0)
            # This means that the identity can be written down as:
            #  A @ B @ A^T = A^T .* [ B @ A^T ]							
            # For the covariance note that: [K_xz K_zz_inv] S [K_xz K_zz_inv]^T = [K_zz_inv K_zx]^T S [K_zz_inv K_zx] =
            # where the output of the linear solver is sol = [K_zz_inv K_zx]. So we have: sol^T S sol. Thus we have: sum(sol.*[S @ sol],0) to compute the diagonal
            # note that as the operations are batched we have to reduce dimension 1 instead of dimension 0. Also use matmul to perform the batched operation.

        # sol = K_zz^{-1}@K_zx
        sol = torch.cholesky_solve(K_zx, L_zz, upper = False)

        if self.is_whiten:
            # cov_qf = K_{xx} -K_{xz} K_{zz}^{-1} K_{zx} + K_{xz} {L_{zz}^T}^{-1} S L_{zz}^{-1}K_{zx} 
            rhs,_ = torch.triangular_solve(K_zx, L_zz, upper = False)
            if diagonal:
                cov_q_f = K_xx - torch.sum(torch.mul(K_zx,sol),1) + torch.sum(torch.mul(rhs,torch.matmul(K_q_U,rhs)),1)
            else:
                cov_q_f = K_xx - torch.matmul(K_xz,sol) + torch.matmul(torch.matmul(torch.transpose(rhs,1,2),K_q_U),rhs)

        else:
            # cov_qf = K_{xx} -K_{xz} K_{zz}^{-1} K_{zx} + [K_{xz} K_{zz}^{-1}] S [K_{xz} K_{zz}^{-1}]^T 
            if diagonal:
                cov_q_f = K_xx - torch.sum(torch.mul(K_zx,sol),1) + torch.sum(torch.mul(sol,torch.matmul(K_q_U,sol)),1)
            else:
                cov_q_f = K_xx - torch.matmul(K_xz,sol) + torch.matmul(torch.matmul(torch.transpose(sol,1,2),K_q_U),sol)

        if diagonal:
            cov_q_f = torch.unsqueeze(cov_q_f,2)

        return mu_q_f, cov_q_f

    def KLD(self) -> torch.tensor :
            """ Kullback Lieber Divergence between q(U) and p(U) 
                Computes KLD of all the GPs at a layer.
                Returns shape (Dy,) with Dy number of outputs GP
            """
            ## whitened representation of inducing points.
            # -> computations got from https://arxiv.org/pdf/2003.01115.pdf

            if self.is_whiten:

                q_U    = self.q_U
                m_q_V  = q_U.variational_mean
                K_q_V  = q_U.chol_variational_covar

                # Variational mean
                m_q_V = m_q_V.view(self.out_dim,self.M,1)

                # Cholesky decomposition of K_vv
                lower_mask = torch.ones(K_q_V.shape[-2:], dtype=cg.dtype, device=cg.device).tril(0)
                L_q_V = K_q_V.mul(lower_mask)
                K_q_V = torch.matmul( L_q_V,L_q_V.transpose(1,2) )

                # KLD
                dot_mean = torch.matmul(m_q_V.transpose(1,2),m_q_V).squeeze()
                log_det_K_q_v = torch.log(torch.diagonal(L_q_V, dim1 = 1, dim2 = 2)**2).sum(1)

                #edit over comment: only true for diagonal matrix
                trace = torch.diagonal(K_q_V,dim1=-2,dim2=-1).sum(-1)

                KLD = 0.5*(-log_det_K_q_v + dot_mean + trace - float(self.M))

            else:
                Z = self.Z
                if self.Z_is_shared:
                    Z = Z.repeat(self.out_dim,1,1)

                ## Posterior
                q_U = self.q_U() # This call generates a torch.distribution.MultivaraiteNormal distribution with the parameters 
                                 # of the variational distribution given by:
                                 # q_mean_U = self.q_U.variational_mean
                                 # q_K_U    = self.q_U.chol_variational_covar


                ## Prior p(U)
                p_mean_U = gpy.lazy.delazify(self.mean_function(Z)).squeeze(-1)
                p_K_U    = self.covariance_function(self.Z, are_equal = False).evaluate() # are_equal = False. We dont add noise to the inducing points, only samples X 
                # shapes (Dy,M,1) and (Dy,M,M)
                #p_U = td.multivariate_normal.MultivariateNormal(p_mean_U,p_K_U)
                p_U = add_jitter_MultivariateNormal(p_mean_U, p_K_U)

                ## KLD -> use built in td.distributions
                KLD = kl_divergence(q_U,p_U)

            return KLD

    def predictive_distribution(self,X: torch.tensor, diagonal: bool=True, S_MC_NNet: int = None)-> list:
        """ This function computes the moments 1 and 2 from the predictive distribution. 
            It also returns the posterior mean and covariance over latent functions.

            p(Y*|X*) = \int p(y*|G(f*)) q(f*,f|u) q(u) df*,df,du
   
                # Homoceodastic Gaussian observation model p(y|f)
                # GP variational distribution q(f)
                # G() represents a non-linear transformation

                Args:
                        `X`                (torch.tensor)  :->: input locations where the predictive is computed. Can have shape (MB,Dx) or (Dy,MB,Dx)
                        `diagonal`         (bool)          :->: if true, samples are drawn independently. For the moment is always true.
                        `S_MC_NNet`        (int)           :->: Number of samples from the dropout distribution is fully_bayesian is true

                Returns:
                        `m1`       (torch.tensor)  :->:  Predictive mean with shape (Dy,MB)
                        `m2`       (torch.tensor)  :->:  Predictive variance with shape (Dy,MB). Takes None for classification likelihoods
                        `mean_q_f` (torch.tensor)  :->:  Posterior mean of q(f) with shape (Dy,MB,1) [same shape as returned by marginal_variational_qf]
                        `cov_q_f`  (torch.tensor)  :->:  Posterior covariance of q(f) with shape (Dy,MB,1) [same shape as returned by marginal_variational_qf]

        """
        if len(X.shape) == 2:
            X = X.repeat(self.out_dim,1,1)
        assert len(X.shape) == 3, "Bad input specificaton"

        self.eval() # set parameters for eval mode. Batch normalization, dropout etc
        if self.fully_bayesian:
            # activate dropout if required
            is_dropout = enable_eval_dropout(self.modules())
            assert is_dropout, "You set the model to work on fully bayesian but there are no dropout layers in your model. I assert this error as otherwise the the code will work in non_bayesian operating mode"

            assert S_MC_NNet is not None, "The default parameter S_MC_NNet is not provided and set to default None, which is invalid for self.be_bayesian" 

        with torch.no_grad():
            if not diagonal:
                raise NotImplemented("This function does not support returning the predictive distribution with correlations")

            mean_q_f, cov_q_f = self.marginal_variational_qf_parameters(X, diagonal = True, is_duvenaud = False, init_Z = None)

            if self.fully_bayesian: # @NOTE: this has not been refactored as with the rest of the code. But note that we could do both point estimate and bayesian by setting S_MC_NNet = 1 for the non
                                    #  bayesian case.
                # If it is fully Bayesian then do it as in the DGP with flows in the output layer
                Dy,MB,_ = mean_q_f.shape

                # 1. Reshape mean_q_f and cov_q_f to shape (Dy,S_MC_NNet*MB)
                mean_q_f_run = mean_q_f.view(Dy,MB).repeat(1,S_MC_NNet)
                cov_q_f_run  = cov_q_f.view(Dy,MB).repeat(1,S_MC_NNet)

                # 2. Compute moments of each of the montecarlos. Just need to provide X extended to S_MC so that each forward computes a monte carlo
                X = X.repeat(1,S_MC_NNet,1) # expand to shape (Dy,S*MB,Dx). 
                MOMENTS = self.likelihood.marginal_moments(mean_q_f_run, cov_q_f_run, self.G_matrix, X) # get the moments of each S*MB samples

                # 3. Compute the moments from the full predictive distribution, e.g the mixture of Gaussians for Gaussian Likelihood
                if isinstance(self.likelihood,GaussianNonLinearMean):
                    m_Y,C_Y = MOMENTS
                    m_Y = m_Y.view(Dy,S_MC_NNet,MB)
                    C_Y = C_Y.view(Dy,S_MC_NNet,MB)

                    m1 = m_Y.mean(1)
                    m2 = ( C_Y + m_Y**2 ).mean(1) - m1**2 # var = 1/S * sum[K_Y + mu_y^2 ] -[1/S sum m1]^2

                elif isinstance(self.likelihood,MulticlassCategorical) or isinstance(self.likelihood,Bernoulli):
                    m1,m2 = MOMENTS,None
                        
                    m1 = m1.view(S_MC_NNet,MB,Dy)
                    m1 = m1.mean(0) # reduce the monte carlo dimension

                else:
                    raise ValueError("Unsupported likelihood [{}] for class [{}]".format(type(self.likelihood),type(self)))

            else:

                MOMENTS = self.likelihood.marginal_moments(mean_q_f.squeeze(dim = 2), cov_q_f.squeeze(dim = 2), diagonal = True, flow = self.G_matrix, X = X) # diagonal True always. Is an element only used by the sparse_MF_GP with SVI. Diag = False is used by standard GP's marginal likelihood

                if isinstance(self.likelihood,GaussianLinearMean) or isinstance(self.likelihood,GaussianNonLinearMean):
                    m1,m2 = MOMENTS 
                elif isinstance(self.likelihood,MulticlassCategorical) or isinstance(self.likelihood, Bernoulli):
                    m1,m2 = MOMENTS,None

        self.train() # switch back to train mode. 
        return m1,m2, mean_q_f, cov_q_f


    ## ====================================== ##
    ## == FINISH MODEL COMPUTATION METHODS == ## 
    ## ====================================== ##


    ## ====================== ##
    ## == TRAINING METHODS == ## 
    ## ====================== ##

    def ELBO(self,X: torch.tensor, Y: torch.tensor) -> torch.tensor:
        """ Evidence Lower Bound

            ELBO = \int log p(y|f) q(f|u) q(u) df,du -KLD[q||p] 

                Args:
                        `X` (torch.tensor)  :->:  Inputs
                        `Y` (torch.tensor)  :->:  Targets

            Returns possitive loss, i.e: ELBO = LLH - KLD; ELL and KLD

        """
        if len(X.shape) == 2:
            X = X.repeat(self.out_dim,1,1) # repeat here as if not this operation will be done twice by the marginal_qf_parameter and likelihood to work batched and multioutput respectively
        assert len(X.shape) == 3, 'Invalid input X.shape' 

        ## ============================= ##
        ## === Compute KL divergence === ##

        KLD = self.KLD()

        ## ================================================= ##
        ## === Computes Variational posterior parameters === ##
        mean_q_f, cov_q_f = self.marginal_variational_qf_parameters(X, diagonal = True, is_duvenaud = False, init_Z = None)
            # mean_q_f shape (Dy,MB,1)
            # cov_q_f  shape (Dy,MB,1)

        ## =============================== ##
        ## Compute Expected Log Likelihood ##
        ELL = self.ELL(X = X, Y = Y, mean = mean_q_f, cov = cov_q_f)

        ## =============================== ##
        ## ======== Computes ELBO ======== ##
        ELL = ELL.sum()
        KLD = KLD.sum()

        ELBO = ELL - KLD 

        ## Accumulate Loss
        KLD  = KLD # Return Possitive KLD

        return ELBO, ELL, KLD # returns positive ELBO. This will be change to negative to optimize


    def ELL(self, X: torch.tensor, Y: torch.tensor, mean: torch.tensor, cov: torch.tensor):
        """ Expected Log Likelihood w.r.t to Gaussian distribution given a non linear mean p(y|G(f))q(f)
            
            ELL = \int log p(y|G(f)) q(f|u) q(u) df,du

                Args:
                        `X`    (torch.tensor)  :->: Inputs, shape  (MB,Dx) or (Dy,MB,Dx)
                        `Y`    (torch.tensor)  :->: Targets, shape (MB,Dy)
                        `mean` (torch.tensor)  :->: Mean from q(f). Shape (Dy,MB,1)
                        `cov`  (torch.tensor)  :->: diagonal covariance from q(f). Shape (Dy,MB,1)

            Computes the stochastic estimator of the ELBO properly re-normalized by N/MB with N number of training points and MB minibatch.

        """
        ## ================ ##
        ## == Assertions == ##
        # Most of the methods of this class can be re-used by any flow base GP. However, the way in which the ELL handles the transformed GP requires specific task likelihoods.
        # In this case, each of the GP is assumed to be the mean of an independent Dy dimensional output Y. The user can simply overwrite the method ELL and the assertions
        # at this point to handle other tasks.
        assert isinstance(self.likelihood,GaussianNonLinearMean) or isinstance(self.likelihood,MulticlassCategorical) or isinstance(self.likelihood,Bernoulli) or isinstance(self.likelihood,GaussianLinearMean), "The current sparse_MF_SP only supports GaussianNonLinearMean likelihood and classification likelihoods. Overwrite this method to perform other tasks and pass the adequate likelihood."

        N,MB = self.N, Y.size(0)
        ELL = self.likelihood.expected_log_prob(Y.t(), mean.squeeze(dim = 2), cov.squeeze(dim = 2), flow = self.G_matrix, X = X)

        return self.N/MB * ELL

    ## ============================= ##
    ## == FINISH TRAINING METHODS == ## 
    ## ============================= ##


    ## ========================= ##
    ## == PERFORMANCE METHODS == ## 
    ## ========================= ##

    def test_log_likelihood(self, X: torch.tensor, Y:torch.tensor, return_moments:bool ,Y_std: float, S_MC_NNet: int = None) -> torch.tensor:
        """ Computes Predictive Log Likelihood 
                \log p(Y*|X*) = \log \int p(y*|G(f*),C_y) q(f*,f|u) q(u) df*,df,du 
                   -> We take diagonal of C_Y as samples are assumed to be i.i.d
                   -> Integration can be approximated either with Monte Carlo or with quadrature. This function uses quadrature.
                
                Args:
                        `X`                 (torch.tensor) :->: Input locations. Shape (MB,Dx) or shape (Dy,MB,Dx)
                        `Y`                 (torch.tensor) :->: Ground truth labels. Shape (MB,Dy)
                        `return_moments`    (bool)         :->: If true, then return the moments 1 and 2 from the predictive distribution.
                        `Y_std`             (float)        :->: Standard desviation of your regressed variable. Used to re-scale output.
                        `S_MC_NNet`         (int)          :->: Number of samples from the dropout distribution is fully_bayesian is true

                Returns:
                        `log_p_y`           (torch.tensor) :->: Log probability of each of the outpus with a tensor of shape (Dy,)
                        `predictive_params` (list)         :->: if return_moments True then returns a list with mean and variance from the predictive distribution. This is done in this funciton
                                                                because for some test log likelihood we need to compute the predictive. Hence support is given for any likelihood. Moments have shape
                                                                (Dy,MB,1)
        """
        MB = X.size(0)
        Dx = X.size(1)
        Dy = self.out_dim
        
        X_run  = X  # didnt realized the rest of function used X_run, so it is easier to do it here.
        if len(X_run.shape) == 2:
            X_run = X_run.repeat(self.out_dim,1,1) 
        assert len(X_run.shape) == 3, 'Invalid input X.shape'

        self.eval() # set parameters for eval mode. Batch normalization, dropout etc
        if self.fully_bayesian:
            # activate dropout if required
            is_dropout = enable_eval_dropout(self.modules())
            assert is_dropout, "You set the model to work on fully bayesian but there are no dropout layers in your model. I assert this error as otherwise the the code will work in non_bayesian operating mode"
            assert S_MC_NNet is not None, "The default parameter S_MC_NNet is not provided and set to default None, which is invalid for self.be_bayesian" 

        with torch.no_grad():

            ## ================================================ ##
            ## =========== GAUSSIAN LIKELIHOOOD =============== ##
            ## == with non linear mean
            if isinstance(self.likelihood,GaussianNonLinearMean):
                # retrieve the noise and expand
                log_var_noise = self.likelihood.log_var_noise
                if self.likelihood.noise_is_shared:
                    log_var_noise = self.likelihood.log_var_noise.expand(Dy,1)

                ## ================================================== ##
                ## === Compute moments of predictive distribution === ##
                #  In this model this is not necessary to compute log likelihood.
                #  However, we give the option of returning this parameters to be consistent
                #  with the standard GP.
                predictive_params = None
                if return_moments:
                    m1,m2, mean_q_f, cov_q_f = self.predictive_distribution(X_run, diagonal = True, S_MC_NNet = S_MC_NNet)
                    predictive_params = [m1,m2]
                else:
                    mean_q_f, cov_q_f = self.marginal_variational_qf_parameters(X_run, diagonal = True, is_duvenaud = False, init_Z = None)
                mean_q_f, cov_q_f = mean_q_f.squeeze(dim = -1),cov_q_f.squeeze(dim = -1)

                self.eval()
                if self.fully_bayesian:
                    ## Call again self.eval() as self.predictive_distribution call self.train() before return
                    is_dropout = enable_eval_dropout(self.modules())
                    assert is_dropout, "You set the model to work on fully bayesian but there are no dropout layers in your model. I assert this error as otherwise the the code will work in non_bayesian operating mode"

                ## Common functions used by bayesian and non bayesian flows
                def get_quad_weights_shifted_locations(mean_q_f,cov_q_f):
                    ## Get the quadrature points and the weights
                    locations = self.likelihood.quadrature_distribution.locations
                    locations = _pad_with_singletons(locations, num_singletons_before=0, num_singletons_after = mean_q_f.dim())
                    shifted_locs = torch.sqrt(2.0 * cov_q_f) * locations + mean_q_f # Shape (S_quad,Dy,S,MB)

                    weights = self.likelihood.quadrature_distribution.weights
                    weights = _pad_with_singletons(weights, num_singletons_before=0, num_singletons_after = shifted_locs.dim() - 1) # Shape (S_quad,1,1,1)

                    return shifted_locs, weights

                def compute_log_lik(Y,Y_std,shifted_locs,C_Y):
                    ## Re-scale by Y_std same as what other people does to compare in UCI
                    Y   = Y_std*Y
                    m_Y = Y_std*shifted_locs
                    C_Y = (Y_std*torch.sqrt(C_Y))**2

                    log_p_y = batched_log_Gaussian( Y, m_Y, C_Y, diagonal = True, cov_is_inverse = False) # (S_quad,Dy,S_MC,MB)
                    
                    return log_p_y

                S_MC_NNet = 1 if not self.fully_bayesian else S_MC_NNet # Note that the estimator is the same for input dependent and Bayesian. Just need to expand or not this dimension
                                                                        
                S_quad = self.quad_points 
                G_mat  = self.G_matrix

                # noise retrieve and reshape
                C_Y = torch.exp(log_var_noise).expand(-1,MB).view(Dy,1,MB,1).repeat((S_quad,1,S_MC_NNet,1,1)) # (Squad,Dy,S_MC_NNet,MB,1). Add extra dimension 1 so that we can compute 
                                                                                                                  #                           likelihood using batched_log_gaussian function    
                # observation reshape
                Y = Y.t().view(1,Dy,1,MB,1).repeat((S_quad,1,S_MC_NNet,1,1))   # S,Dy,S_MC_NNet,MB,1

                # Y_std reshape
                Y_std = Y_std.view(1,Dy,1,1,1).repeat(S_quad,1,S_MC_NNet,MB,1) # S,Dy,S_MC_NNet,MB,1

                # this operation could be done by repeating X and computing mean_q_f as in DGP but is not necessary to do extra computation here as X is constant: just repeat. 
                mean_q_f, cov_q_f = mean_q_f.unsqueeze(dim = 1),cov_q_f.unsqueeze(dim = 1) # Remove last dimension, so that we can warp. We add it later for batched_log_lik
                mean_q_f = mean_q_f.repeat(1,S_MC_NNet,1) # (Dy,S_MC_NNet,MB)
                cov_q_f  = cov_q_f.repeat(1,S_MC_NNet,1)

                ## =================================== ##
                ## === Compute test log likelihood === ##
                shifted_locs, weights =  get_quad_weights_shifted_locations(mean_q_f,cov_q_f)

                ## Warp quadrature points
                # expand X to perform MC dropout over NNets parameters
                X_run = X_run.unsqueeze(dim = 1).repeat(1,S_MC_NNet,1,1) # Just add one extra dimension. No need for repeat for S_quad as pytorch automatically broadcasts. 
                                                                         # It is important to repeat over S_MC_NNet. In this way each forward thorugh X computes a different 
                                                                         # MC for the flow parameters. Otherwise pytorch would broadcast S_MC_NNet as well hence we would only 
                                                                         # be using one sample from the posterior over W.
                for idx,fl in enumerate(G_mat):
                     shifted_locs[:,idx,:,:] = fl(shifted_locs[:,idx,:,:],X_run[idx]) # (S_quad,Dy,S_MC_NNet,MB)

                shifted_locs = shifted_locs.view(S_quad,Dy,S_MC_NNet,MB,1) # shape (S_quad,Dy,S,MB,1)

                log_p_y = compute_log_lik(Y,Y_std,shifted_locs,C_Y)

                if self.fully_bayesian: # the only difference between bayesian and the rest is here, where we perform a double integration for this case

                    # Reduce with double logsumexp operation. Check estimator here: @TODO: add link once we releasea github
                    reduce_lse = torch.log(weights)  + log_p_y
                    log_p_y = torch.logsumexp( torch.logsumexp(reduce_lse, dim = 0) -0.5*torch.log(cg.pi) ,dim = 1).sum(1) - MB*numpy.log(S_MC_NNet)
                else:
                    # Note that we just need to remove the extra dimension we added for using the same code
                    log_p_y = log_p_y.squeeze(dim = 2)
                    weights = weights.squeeze(dim = 2)
        
                    ## Reduce log ws + log_p_y_s using logsumexp trick. Also reduce MB and add the constant
                    reduce_lse = torch.log(weights) + log_p_y
                    log_p_y = (torch.logsumexp(reduce_lse, dim = 0)).sum(-1) - 0.5*MB*torch.log(cg.pi)

            ## ===================
            ## == with linear mean
            elif isinstance(self.likelihood,GaussianLinearMean):
                ## ================================================== ##
                ## === Compute moments of predictive distribution === ##
                m_Y,K_Y, mean_q_f, cov_q_f = self.predictive_distribution(X_run, diagonal = True)

                ## =================================== ##
                ## === Compute test log likelihood === ##
                # Re-scale Y_std
                Y = Y.t() # (Dy,MB)
                Y_std = Y_std.view(self.out_dim,1) # (Dy,1)

                log_p_y = batched_log_Gaussian( obs = Y_std*Y, mean = Y_std*m_Y, cov = (Y_std*torch.sqrt(K_Y))**2, diagonal = True, cov_is_inverse = False)

                predictive_params = None
                if return_moments:
                    predictive_params = [m_Y,K_Y]

            ## =============================================================== ##
            ## ============ BERNOULLI/CATEGORICAL LIKELIHOOOD ================ ##
            elif isinstance(self.likelihood,MulticlassCategorical) or isinstance(self.likelihood,Bernoulli):

                # as we cant do exact integration here either we warp or we dont the proceedure is very similar to GP classification. The only difference is of
                # binary classification with Gauss CDF link function
                m_Y, _, mean_q_f, cov_q_f = self.predictive_distribution(X_run,diagonal = True, S_MC_NNet = S_MC_NNet)

                check = torch.logical_not(torch.isfinite(m_Y)).float()
                assert check.sum() == 0.0, "Got saturated probabilities"

                if isinstance(self.likelihood,Bernoulli): # turn the vector as if it became from the MulticlassCategorical so that this is transparent to the trainer
                    m_Y     = m_Y.squeeze() 
                    neg_m_Y = 1.0-m_Y # compute the probability of class 0
                    m_Y     = torch.stack((neg_m_Y,m_Y),dim = 1) 

                _, _ , _ , log_p_y = compute_calibration_measures(m_Y.float() ,Y ,apply_softmax = False ,bins = 15)  

                log_p_y = -1*((log_p_y*MB).sum()) # the compute_calibration_measures returns log_p_y.mean(), hence we remove that by multiplying by MB and then summing up

                predictive_params = None
                if return_moments:
                    predictive_params = [m_Y]

            else:
                raise ValueError("Unsupported likelihood [{}] for class [{}]".format(type(self.likelihood),type(self)))

        self.train() # set parameters for train mode. Batch normalization, dropout etc
        return log_p_y, predictive_params


