# torch
import torch
import torch.nn as nn

# Python
import sys
sys.path.extend(['../../../'])

# Custom
import config as cg

class FC_VI_LR(nn.Module):
    def __init__(self,indim,outdim,batch,activation,prior_mean,prior_logvar):
        super(FC_VI_LR, self).__init__()
        ## Model Definitinion
        # -> Prior: p(w,b)=Normal(0,1)
        self.activation=nn.functional.relu if activation=='relu' else None
        self.outdim=outdim

        ## Variational Distribution
        self.w_mean=nn.Parameter(torch.randn((indim,outdim)))
        self.w_logvar=nn.Parameter(torch.randn((indim,outdim)))
        self.b_mean=nn.Parameter(torch.randn((outdim)))
        self.b_logvar=nn.Parameter(torch.randn((outdim)))

        # Monitor Model Collapse
        self.eps_mean_upper=prior_mean+0.01
        self.eps_mean_lower=prior_mean-0.01
        self.eps_logvar_upper=prior_logvar+0.01
        self.eps_logvar_lower=prior_logvar-0.01

        self.prior_mean=prior_mean
        self.prior_logvar = prior_logvar


    def sample(self,ind_mu,ind_var):
        # Reparameterization trick ( locally )
        vec = torch.zeros_like(ind_mu).to(cg.device).normal_()
        s=(vec*(ind_var.sqrt()) + ind_mu)
        return s

    def forward(self,x):
        # Induce distribution over activations
        ind_mu = torch.matmul(x,self.w_mean) + self.b_mean
        ind_var = torch.matmul(x**2,torch.exp(self.w_logvar)) + torch.exp(self.b_logvar)
        # Sample the the batch of activations
        s=self.sample(ind_mu,ind_var)
        return self.activation(s) if self.activation != None else s

    def forward_without_LR(self,x):

        assert len(x.shape) == 2, "This function does not work batched, need to implement"

        w_m = self.w_mean
        w_v = torch.exp(self.w_logvar)

        b_m = self.b_mean
        b_v = torch.exp(self.b_logvar)

        w  = self.sample(w_m,w_v)
        b  = self.sample(b_m,b_v)

        s  = torch.mm(x,w) + b

        return self.activation(s) if self.activation != None else s

    def get_total_params(self):
        # Return total number of parameters
        return self.w_mean.numel()*2+self.b_mean.numel()*2

    def get_collapsed_posterior(self):
        # Check If the parameters has collapsed to the prior
        w=((self.w_mean<=self.eps_mean_upper) & (self.w_mean>=self.eps_mean_lower) & (self.w_logvar<=self.eps_logvar_upper) & (self.w_logvar>=self.eps_logvar_lower)).float().sum()
        b=((self.b_mean<=self.eps_mean_upper) & (self.b_mean>=self.eps_mean_lower) & (self.b_logvar<=self.eps_logvar_upper) & (self.b_logvar>=self.eps_logvar_lower)).float().sum()

        return w+b

    def get_KLcollapsed_posterior(self):
        # Check If the parameters has collapsed to the prior
        w_kl = 0.5 * (torch.exp(self.w_logvar - self.prior_logvar)
              + (self.w_mean - self.prior_mean)**2/torch.exp(self.prior_logvar)
              - 1 + (self.prior_logvar - self.w_logvar))

        w = (w_kl <= 7.5e-05).sum()
        b_kl = 0.5 * (torch.exp(self.b_logvar - self.prior_logvar)
              + (self.b_mean - self.prior_mean)**2/torch.exp(self.prior_logvar)
              - 1 + (self.prior_logvar - self.b_logvar))

        b = (b_kl <= 7.5e-05).sum()

        return w+b


## Mean-field Gaussian VI over a BNN with Local Reparameterization

class BNN_VILR(nn.Module):
    def __init__(self,dim_layer,number_layers,in_dim,out_dim,prior_mean,prior_var,batch,dataset_size, use_batched_computations):
        super(BNN_VILR, self).__init__()

        ## General Parameters
        # likelihood p(t|x)
        self.ce=nn.functional.cross_entropy
        self.softmax=nn.functional.softmax

        # prior N(w,b| )
        self.prior_mean=torch.tensor(prior_mean).to(cg.device)
        self.prior_logvar=torch.log(torch.tensor(prior_var).to(cg.device))

        # Loss utils
        self.N = dataset_size

        # Instance model Likelihood
        self.Layers = nn.ModuleList()
        if number_layers == 0:
            Lin=FC_VI_LR(in_dim,out_dim,batch,'linear',self.prior_mean,self.prior_logvar)
            self.Layers.append(Lin)
        else:
            Lin=FC_VI_LR(in_dim,dim_layer,batch,'relu',self.prior_mean,self.prior_logvar)
            self.Layers.append(Lin)
            for l in range(number_layers-1):
                self.Layers.append(FC_VI_LR(dim_layer,dim_layer,batch,'relu',self.prior_mean,self.prior_logvar))
            self.Layers.append(FC_VI_LR(dim_layer,out_dim,batch,'linear',self.prior_mean,self.prior_logvar))

        # Performance utils
        self.use_batched_computations = use_batched_computations # all the computations are performed in parallel

    def __get_collapsed_posterior__(self):
        # get the percentage of collapsed parameters
        with torch.no_grad():
            klcollaps,collaps,total_params=[0.0]*3
            for l in self.Layers:
                klcollaps+=l.get_KLcollapsed_posterior()
                collaps+=l.get_collapsed_posterior()
                total_params+=l.get_total_params()
            return (klcollaps/float(total_params)*100., collaps/float(total_params)*100.)

    # Compute the likelihood p(t|x)
    def forward(self,x, with_LR = True):
        if with_LR:
            for l in self.Layers:
                x = l(x)
        else:
            for l in self.Layers:
                x = l.forward_without_LR(x)
        return x

    # Gaussian KLD
    def GAUSS_KLD(self,qmean,qlogvar,pmean,plogvar):
        # Computes the DKL(q(x)//p(x)) between the variational and the prior
        # distribution assuming Gaussians distribution with arbitrary prior
        qvar,pvar = torch.exp(qlogvar),torch.exp(plogvar)
        DKL=(0.5 * (-1 + plogvar - qlogvar + (qvar/pvar) + torch.pow(pmean-qmean,2)/pvar)).sum()

        return DKL

    # Kullback Liber Divergence of the Full Model
    def KLD(self):
        DKL=0.0
        for l in self.Layers:
            w_mean,w_logvar,b_mean,b_logvar=l.w_mean,l.w_logvar,l.b_mean,l.b_logvar
            DKL+=(self.GAUSS_KLD(w_mean,w_logvar,self.prior_mean,self.prior_logvar)+self.GAUSS_KLD(b_mean,b_logvar,self.prior_mean,self.prior_logvar))

        return DKL


    # Evidence Lower Bound
    def ELBO(self,x,t,MC_samples):

        MB = x.size(0) # mini batch
        NLLH = 0.0 #torch.zeros((MB,)).to(device)

        # =======================
        # Negative Log Likelihood

        if self.use_batched_computations:
            # it is always better to use batched computations, although this might run out of memory

            x    = x.unsqueeze(dim = 0).repeat(MC_samples,1,1)
            t    = t.unsqueeze(dim = 0).repeat(MC_samples,1)

            NLLH = self.ce( self.forward(x).view(MC_samples*MB,-1), t.view(MC_samples*MB) , reduction = 'none').view(MC_samples,MB)
            NLLH = NLLH.mean(0) # reduce the monte carlo dimension

        else:
            for mc in range(MC_samples): #stochastic likelihood estimator
                # Reduction = None. We will perform the reduction to propely
                # scale the two terms in the ELBO
                NLLH+= self.ce(self.forward(x),t, reduction = 'none')

            NLLH /= float(MC_samples)

        NLLH =  NLLH.sum()
        NLLH *= float(self.N)/float(MB) # re-scale by minibatch size

        # =============
        # Possitive KLD
        DKL=self.KLD()

        # =========================
        # -ELBO = -log p(t|x) + DKL
        ELBO = NLLH + DKL
        return ELBO,NLLH,DKL

    # Train function
    def train(self,x,t,scheduler,epochs,lr,warm_up, MC_samples):

        optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        # Adam goes better for this kind of models than SGD, eventhough is not
        # a correct optimizer, see https://openreview.net/pdf?id=ryQu7f-RZ .
        # However It always works fine for models based on
        # reparametrization trick.

        for e in range(epochs):
            # optimizer
            lr_=scheduler(lr,e)

            for idx in range(len(optimizer.param_groups)):
                optimizer.param_groups[idx]['lr'] = lr_

            loss,NLLH,KL=self.ELBO(x,t,MC_samples)
            # warm up (see https://arxiv.org/abs/1602.02282)
            loss = loss if e > warm_up else NLLH

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            KLcollapsed_posterior_percentage, collapsed_posterior_percentage=self.__get_collapsed_posterior__()
            print("On epoch {} LR {:.3f} ELBO {:.5f} NNL {:.5f} KL {:.5f} COLLAPSED PARAMS {:.3f}% KLCOLLAPSED PARAMS {:.3f}%".format(e,lr_,loss.item(),NLLH.item(),KL.item(),collapsed_posterior_percentage.item(), KLcollapsed_posterior_percentage.item()))


    # Compute the predictive distribution
    def predictive(self,x,samples, use_same_samples = False):
        # Compute predictive distribution
        # We use a different w per test sample for the montecarlo integrator.
        # It does not matter. Normally I perform inference in a different way
        # by using the same set of parameters for all the test samples. To allow that
        # set use_same_samples = True
        with torch.no_grad():

            if self.use_batched_computations:
                x          = x.unsqueeze(dim = 0).repeat(samples,1,1)
                prediction = self.softmax(self.forward(x, with_LR = not use_same_samples),dim=2)
                return prediction.mean(0)

            else:
                prediction=0.0
                for s in range(samples):
                    prediction+=self.softmax(self.forward(x, with_LR = not use_same_samples),dim=1)

                return prediction/float(samples)
