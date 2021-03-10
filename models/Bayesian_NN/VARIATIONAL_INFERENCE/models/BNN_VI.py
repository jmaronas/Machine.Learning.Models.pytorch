# torch
import torch
import torch.nn as nn

# Python
import sys
sys.path.extend(['../../../'])

# Custom
import config as cg

## Layer class for mean field Gaussian VI BNN in Fully conected Network
class FC_VI(nn.Module):
    def __init__(self,indim,outdim,activation,prior_mean,prior_logvar):
        super(FC_VI, self).__init__()
        ### Model Definitinion  ###
        # -> Prior: p(w,b)=Normal(0,1)
        self.activation=nn.functional.relu if activation=='relu' else None
        self.outdim=outdim

        ## Variational Distribution
        self.w_mean=nn.Parameter(torch.randn((indim,outdim)))
        self.w_logvar=nn.Parameter(torch.randn((indim,outdim)))
        self.b_mean=nn.Parameter(torch.randn((outdim)))
        self.b_logvar=nn.Parameter(torch.randn((outdim)))

        ## Utils
        self.sampler_w=torch.zeros((indim,outdim)).to(cg.device)
        self.sampler_b=torch.zeros((outdim,)).to(cg.device)

        # Monitor Mode Collapse
        self.eps_mean_upper=prior_mean+0.01
        self.eps_mean_lower=prior_mean-0.01
        self.eps_logvar_upper=prior_logvar+0.01
        self.eps_logvar_lower=prior_logvar-0.01

        self.prior_mean=prior_mean
        self.prior_logvar = prior_logvar

    def sample(self):
        # Reparameterization trick
        w_s=(self.sampler_w.normal_().clone().detach())*torch.exp(0.5*self.w_logvar) + self.w_mean
        b_s=(self.sampler_b.normal_().clone().detach())*torch.exp(0.5*self.b_logvar) + self.b_mean
        return w_s,b_s

    def forward(self,x):
        # Forward by previously sampling using reparaemterization
        w,b=self.sample()
        return self.activation(torch.mm(x,w)+b) if self.activation != None else torch.mm(x,w)+b

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


## Mean-field Gaussian VI over a BNN
class BNN_VI(nn.Module):
    def __init__(self,dim_layer,number_layers,in_dim,out_dim,prior_mean,prior_var,dataset_size, use_batched_computations):
        super(BNN_VI, self).__init__()

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
            Lin=FC_VI(in_dim,out_dim,'linear',self.prior_mean,self.prior_logvar)
            self.Layers.append(Lin)
        else:
            Lin=FC_VI(in_dim,dim_layer,'relu',self.prior_mean,self.prior_logvar)

            self.Layers.append(Lin)
            for l in range(number_layers-1):
                self.Layers.append(FC_VI(dim_layer,dim_layer,'relu',self.prior_mean,self.prior_logvar))
            self.Layers.append(FC_VI(dim_layer,out_dim,'linear',self.prior_mean,self.prior_logvar))

        # For efficient computation
        self.use_batched_computations = use_batched_computations
        assert self.use_batched_computations == False, "Not implemented"

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
    def forward(self,x):
        for l in self.Layers:
            x=l(x)
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
    def predictive(self,x,samples):
        # Draw samples from the predictive using ancestral sampling
        # We use a different w per test sample for the montecarlo integrator.
        # It does not matter. Normally I perform inference in a different way
        # when using local rep, however for simplicity I let this like this
        with torch.no_grad():

            prediction=0.0
            for s in range(samples):
                prediction+=self.softmax(self.forward(x),dim=1)

            return prediction/float(samples)
