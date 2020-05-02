# torch
import torch
import torch.nn as nn

# Python
import sys
sys.path.extend(['../../../'])

# Custom
import config
device = config.device


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

		# Utils
		self.sampler=torch.zeros((batch,outdim)).to(device)

		# Monitor Model Collapse
		self.eps_mean_upper=prior_mean+0.01
		self.eps_mean_lower=prior_mean-0.01
		self.eps_logvar_upper=prior_logvar+0.01
		self.eps_logvar_lower=prior_logvar-0.01

	def __resample__(self,batch):
		# This function reshapes the sampler in case the batch dimension changes
		self.sampler=torch.zeros((batch,self.outdim)).to(device)

	def sample(self,ind_mu,ind_var):
		# Reparameterization trick ( locally )
		s=(self.sampler.normal_().clone().detach()*(ind_var.sqrt()) + ind_mu)
		return s

	def forward(self,x):
		# Induce distribution over activations
		ind_mu = torch.mm(x,self.w_mean) + self.b_mean
		ind_var = torch.mm(x**2,torch.exp(self.w_logvar)) + torch.exp(self.b_logvar)
		# Sample the the batch of activations
		s=self.sample(ind_mu,ind_var)
		return self.activation(s) if self.activation != None else s

	def get_total_params(self):
		# Return total number of parameters
		return self.w_mean.numel()*2+self.b_mean.numel()*2

	def get_collapsed_posterior(self):
		# Check If the parameters has collapsed to the prior
		w=((self.w_mean<=self.eps_mean_upper) & (self.w_mean>=self.eps_mean_lower) & (self.w_logvar<=self.eps_logvar_upper) & (self.w_logvar>=self.eps_logvar_lower)).float().sum()
		b=((self.b_mean<=self.eps_mean_upper) & (self.b_mean>=self.eps_mean_lower) & (self.b_logvar<=self.eps_logvar_upper) & (self.b_logvar>=self.eps_logvar_lower)).float().sum()

		return w+b


## Mean-field Gaussian VI over a BNN with Local Reparameterization

class BNN_VILR(nn.Module):
	def __init__(self,dim_layer,number_layers,in_dim,out_dim,prior_mean,prior_var,batch,dataset_size):
		super(BNN_VILR, self).__init__()

		## General Parameters
		# likelihood p(t|x)
		self.ce=nn.functional.cross_entropy
		self.softmax=nn.functional.softmax

		# prior N(w,b| )
		self.prior_mean=torch.tensor(prior_mean).to(device)
		self.prior_logvar=torch.log(torch.tensor(prior_var).to(device))

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


	def __get_collapsed_posterior__(self):
		# get the percentage of collapsed parameters
		with torch.no_grad():
			collaps,total_params=[0.0]*2
			for l in self.Layers:
				collaps+=l.get_collapsed_posterior()
				total_params+=l.get_total_params()
			return collaps/float(total_params)*100.

	# This method is only needed by this model
	def __resample__(self,batch):
		for l in self.Layers:
			l.__resample__(batch)

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

		MB = x.size(0)# mini batch
		NLLH = 0.0 #torch.zeros((MB,)).to(device)

		# Negative Log Likelihood
		for mc in range(MC_samples): #stochastic likelihood estimator
			# Reduction = None. We will perform the reduction to propely
			# scale the two terms in the ELBO
			NLLH+= self.ce(self.forward(x),t, reduction = 'none')

		NLLH /= float(MC_samples)
		NLLH =  NLLH.sum()
		NLLH *= float(self.N)/float(MB) # re-scale by minibatch size

		# Possitive KLD
		DKL=self.KLD()

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
			optimizer=torch.optim.Adam(self.parameters(),lr=lr_)

			loss,NLLH,KL=self.ELBO(x,t,MC_samples)
			# warm up (see https://arxiv.org/abs/1602.02282)
			loss = loss if e > warm_up else NLLH

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			collapsed_posterior_percentage=self.__get_collapsed_posterior__()
			print("On epoch {} LR {:.3f} ELBO {:.5f} NNL {:.5f} KL {:.5f} COLLAPSED PARAMS {:.3f}%".format(e,lr_,loss.item(),NLLH.item(),KL.item(),collapsed_posterior_percentage.item()))


	# Compute the predictive distribution
	def predictive(self,x,samples):
		# Draw samples from the predictive using ancestral sampling
		# We use a different w per test sample for the montecarlo integrator.
		# It does not matter. Normally I perform inference in a different way
		# when using local rep, however for simplicity I let this like this
		with torch.no_grad():
			self.__resample__(x.size(0))
			prediction=0.0
			for s in range(samples):
				prediction+=self.softmax(self.forward(x),dim=1)

			return prediction/float(samples)
