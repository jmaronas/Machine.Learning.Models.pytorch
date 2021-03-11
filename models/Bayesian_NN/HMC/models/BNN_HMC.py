## Bayesian Neural Network with Hamiltonian Monte Carlo Inference as proposed by Duane and later Radford Neal
## Author: Juan Maroñas Molano
## NOTE: this is not a super efficient implementation of HMC. The intention of this implmentation is to be illustrative of the algorithm, rather than efficient. I try to explicitely point to the parts
##       of the code where extra computation is done. This merely involve the computation of the potential energy which is used in the leap frog integrator and also for the metropolis hasting correction.
##       Basically one can reuse computations done in the leap frog integrator for the metropolis hasting correction and viceversa. This is not done in this code but should be consider if one wants to use
##       very large Neural Nets (obviously one would probably not use this code for that and go for a NUTS implmentation).

# Standard
import numpy

# Python
import sys
sys.path.extend(['../../../'])
import math

# Torch
import torch
import torch.nn as nn

# Custom
import config as cg

class FC(nn.Module):
    def __init__(self,indim,outdim,activation):

        super(FC, self).__init__()

        # Parameters of the MCMC algorithm are initialize to the standard Normal distribution
        self.w = torch.randn((indim,outdim)).to(cg.device)         
        self.b = torch.randn((outdim)).to(cg.device)

        # Momentum parameters (kinetic) associated with parameters (potential energy)
        self.mmu_w=torch.randn(indim,outdim).to(cg.device)
        self.mmu_b=torch.randn((outdim)).to(cg.device)

        # Compute gradients from the parameters that are sampled
        self.w.requires_grad = True
        self.b.requires_grad = True

        # Activation function
        self.activation=nn.functional.relu if activation=='relu' else None

    ## Resample momentum parameters 
    def __resample__(self):
        self.mmu_w.normal_()
        self.mmu_b.normal_()

    ## Forward through the likelihood model
    def forward(self,x):
        return self.activation(torch.mm(x,self.w)+self.b) if self.activation != None else torch.mm(x,self.w)+self.b

    ## Set gradients to zero
    def zero_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()

    ## Initialize the chain parameters using a sample from the standard normal
    def initialize(self):
        self.w.normal_()
        self.b.normal_()

    ## Set parameters to a value ( usefull when the proposal is rejected )
    def set_params(self,w,b):
        self.w = w
        self.b = b

class BNN_HMC(nn.Module):
    def __init__(self,neur,number_layers,input_dim,out_dim,prior_mean,prior_var,L=20, interval_L = 5 , eps=0.01, interval_eps = 0.001):

        super(BNN_HMC, self).__init__()

        ## Initialize the likelihood model
        if number_layers == 0:
            Lin=FC(input_dim,out_dim,'linear')
            self.Layers=[Lin]

        else:
            Lin=FC(input_dim,neur,'relu')
            self.Layers=[Lin]
            for l in range(number_layers-1):
                self.Layers.append(FC(neur,neur,'relu'))

            self.Layers.append(FC(neur,out_dim,'linear'))

        self.ce=nn.functional.cross_entropy
        self.softmax=nn.functional.softmax

        ## Initialize prior distributions from the potential energy p(w)
        self.prior_mean=torch.tensor(prior_mean).to(cg.device)
        self.prior_var=torch.tensor(prior_var).to(cg.device)

        ## Initialize HMC model parameters
        self.leapfrogsteps          = L
        self.interval_leapfrogsteps = interval_L
        self.epsilon                = eps
        self.interval_epsilon       = interval_eps

        self.epsilon = eps
        self.acceptance_prob = 0.0

        ## Some utils
        self.pi=torch.tensor(math.pi).to(cg.device)
        
    def __set_parameters__(self,parameters):
        for layer,(w,b) in zip(self.Layers,parameters):
            layer.set_params(w,b)

    ## Resample momentum parameters from the whole hierarchy
    def __resample__(self):
        for layer in self.Layers:
            layer.__resample__()

    ## Get the current parameters from the chain, so that we can use them if the propsal is rejected
    def __parameters__(self, detach = False):
        a=[]
        for l in self.Layers:
            w, b = l.w, l.b
            if detach:
                w = w.detach()
                b = b.detach()
            a+=[(w,b)]
        return a

    ## Log probability of the prior distribution p(w) used by the potential energy; assumes factorized Gaussian prior
    def __lognormal__(self):
        LPRIOR=0
        for l in self.Layers:
            cte = -0.5*torch.log(2*self.pi) -0.5*torch.log(self.prior_var)
            LPRIOR += ((-1/(2*self.prior_var)*(l.w-self.prior_mean)**2 + cte).sum() + (-1/(2*self.prior_var)*(l.b-self.prior_mean)**2 + cte).sum())

        return LPRIOR

    ## Kinetic Energy from the HMC algorithm
    def __kinetic__(self):
        #computes the kinetic function for all the momentum associated with each parameter
        K=0
        for l in self.Layers:
            K+=0.5*((l.mmu_w**2).sum()+(l.mmu_b**2).sum()) # this assumes mass matrix M = I, otherwise need to do (1/2mi)*mmu 
        return K

    ## Potential Energy from the HMC algorithm  U = -\log p(w|x,y) \propto - log p(t|x,w) - log p(w)
    def __potential__(self,x,t):
        with torch.no_grad(): # we will not compute derivatives just evaluate unnormalized posterior
            logit     = self.forward(x)
            NLOGLH    = self.ce(logit,t,reduction='sum')  # it is important to sum and not apply the 1/N normalization factor. The -loglikelihood=N*Cross_Entropy
            NLOGPRIOR = -1*self.__lognormal__()           # we negate as we need negative log prior: -log P(w) = -log N(0,I)-->__lognormal__ computes N(0,I)
        return NLOGLH+NLOGPRIOR

    ## Compute the Hamiltonian
    def H(self,x,t):
        # compute H=U(w)+K(m)
        # U(w) = -log [\prod p(t_i|x_i,w) \cdot p(w)]
        # K(m) = 0.5 \sum_i^{|P|} m_i^T \cdot m_i
        K = self.__kinetic__()	
        U = self.__potential__(x,t)
        return K+U

    ## Forward through the likelihood model p(t|x)
    def forward(self,x):
        for layer in self.Layers:
            x=layer(x)
        return x

    ## Draw a sample from the posterior distribution. Run the discretization of Hamiltonian equations using Leap frog method. This function could be done much more modular, but keeping things in a same
    #  function to make easier to track for those who want to learn. This function could also be done much more efficient as the evaluation of the Hamiltonian from initial parameters used for 
    #  Metropolis hasting correction could be taken from the previous iteration. Also, we could use this evaluation to directly backpropagate and obtain the gradients for the first step in the
    #  leap frog method. Finally, the potential energy computation of the Hamiltonian of the proposed parameters could be also obtained from the evaluation of the LLH used in the third step in the chain.
    #  The potential energy is the one requiring a forward through the NNet which is what is costly for Deep NNs. Moreover, the gradient steps of the sympletic integrator should be moved into the FC class
    #  , however I keep them in this function to be able to understand what is going on (also the prior on the parameters p(w) should be layer dependent and not globaly dependent), and should not
    #  be computed explicitely so that any desirable prior can be used
    def sample_from_posterior(self,x,t):

        # Resample the momentum variables
        self.__resample__()
        
        # Store the Hamiltonian on the previous parameters for the metropolis hasting correction. Note that this H_prev could be taken from the last iteration. Recomputing it requires a forward
        # through the model which is expensive for deep NNs
        H_prev = self.H(x,t)

        # Get current parameters of the t-th run of the MCMC algorithm in case the proposal is rejected
        init_params = self.__parameters__() 

        # =================================================
        # SIMULATE HAMILTONIAN DYNAMICS USING LEAPFROG STEP

        ## Get the epsilon and the step size by sampling from a uniform distribution. These values are freezed during the whole discretization proceedure
        L            = self.leapfrogsteps          
        interval_L   = self.interval_leapfrogsteps
        eps          = self.epsilon                
        interval_eps = self.interval_epsilon

        leap_frog_step = numpy.random.randint(L-interval_L,L+interval_L+1)  # sample leapfrog step. Done for avoiding loops
        epsilon        = numpy.random.uniform(eps-interval_eps,eps+interval_eps)   # sample epsilon

        ## ========================
        ## ========================
        # 1) compute p' at epsilon/2
        # we perform half step (0.5* \epsilon) to update p

        # first need to get the gradients of the parameters
        logit = self.forward(x)                  # in order to compute the derivative of p(w|\mathcal{O})
        LLH   = self.ce(logit,t,reduction='sum') # -log likelihood. Part of the gradient of the posterior is computed by backpropagating this error. Note that this is equivalent to the cross entropy multiplied by the sample size.
        LLH.backward() # so we have the gradient of the LLog likelihood w.r.t each parameter

        ## see below (NOTE FIRST STEP) why this first step 1) is not kept under the loop for step in range(leap_frog_step) as determined by the algorithm
        for l in self.Layers:
            #for each bias and each w
            #p' = p -0.5 * epsilon * \nabla_w U(w) [ U(w) = - \log p(t|x,w) - log p(w) ]

            gradient_w = l.w.grad.data+(l.w.data-self.prior_mean)/self.prior_var # derivative or CE + derivative of -log normal (w)
            l.mmu_w    = l.mmu_w-self.epsilon/2.*gradient_w                      # perform the step to compute p'

            gradient_b = l.b.grad.data+(l.b.data-self.prior_mean)/self.prior_var # derivative or CE + derivative of -log normal (w)
            l.mmu_b    = l.mmu_b-self.epsilon/2.*gradient_b                           # perform the step to compute p'. We save it in the momentum variable from the class

            l.zero_grad() # this not necessary as the next two lines of code reset grads, however I put it for clarity. The gradients of U here are evaluated at w


        for step in range(leap_frog_step):
        
            ## ========================
            ## ========================
            #2) compute  w_step using the previous p'

            for l in self.Layers:
                # we now update w (q in Neal's report)
                # w_step = w + \epsilon \nabla_p (k(p')) = w + \epsilon * p'  => remember that \nabla_p K(p) = p

                # call data here to use the same l.w variable. If we do l.w= then a new variable is created with requires_grad set to false.
                # this reset grads (so the avobe l.zero_grad() is really not necessary. However this is pytorch specific so I let the above code for clarity
                l.w.data = l.w.data + self.epsilon*l.mmu_w ## note that here we have assume that the mass matrix from the HMC is always M = I, otherwise this update would be w + eps * (1/m_i) * pi
                l.b.data = l.b.data + self.epsilon*l.mmu_b	
                                

            ## ========================
            ## ========================
            #3) compute p' with a full \epsilon step but starting from the previous at 0.5 \epsilon at step=1; or at \epsilon if step>1 

            # We need gradients w.r.t the new w1
            # compute forward with parameter w1

            logit = self.forward(x)
            LLH   = self.ce(logit,t,reduction='sum') 
            LLH.backward()

            for l in self.Layers:

                if (step+1) == self.leapfrogsteps: # in the last step we only move 0.5*\epsilon
                    gradient_w = l.w.grad.data+(l.w.data-self.prior_mean)/self.prior_var # derivative or CE + derivative of -log normal (w)
                    l.mmu_w    = l.mmu_w-self.epsilon/2.*gradient_w # perform the step to compute p'

                    gradient_b = l.b.grad.data+(l.b.data-self.prior_mean)/self.prior_var # derivative or CE + derivative of -log normal (w)
                    l.mmu_b    = l.mmu_b-self.epsilon/2.*gradient_b # perform the step to compute p'. We save it in the momentum variable from the class

                else: # NOTE FIRST STEP: instead of computing the step 1) each time in the leap frog loop, we can move 1 full step epsilon rather than 0.5 so that the result of this step is equal to the
                      # step 3) and the step 1). Only in the last leapfrog step we only move 0.5*epsilon
                    gradient_w = l.w.grad.data+(l.w.data-self.prior_mean)/self.prior_var # derivative or CE + derivative of -log normal (w)
                    l.mmu_w    = l.mmu_w-self.epsilon*gradient_w # perform the step to compute p'

                    gradient_b = l.b.grad.data+(l.b.data-self.prior_mean)/self.prior_var # derivative or CE + derivative of -log normal (w)
                    l.mmu_b    = l.mmu_b-self.epsilon*gradient_b # perform the step to compute p'. We save it in the momentum variable from the class
          
                l.zero_grad()

        if(torch.isnan(LLH)):
            raise(Error('Dynamic simulation saturates'))

        # Metropolist Hasting Correction to eliminate the discretization errors.
        u = numpy.random.uniform()
        alfa = torch.min(torch.tensor(1.0),torch.exp(H_prev-self.H(x,t))) # An efficient implementation of this algorithm would not compute H(x,t) here, as this requires a forward through the method
                                                                          # We could just reuse the computatio of LLH done for step 3, combined with the log p(w) and just a computation of the kinetic
                                                                          # energy with the momentum variables resulting from step 3), which is a cheap step.
                                                                        
        self.acceptance_prob+=1
        if u > alfa:
            self.__set_parameters__(init_params)
            self.acceptance_prob-=1

                
    ## Run the MCMC algorithm
    def run_MCMC(self,x,t,steps,store_all=True,warmup=0,avoidcorrelation=1):
        tot_runs = steps*(avoidcorrelation)+warmup # number of total iterations to perform
        sys.stderr.write("Params samples {} warmup {} thinning {}\n".format(steps,warmup,avoidcorrelation))
        sys.stderr.write("Need to run a total of {} loops\n".format(tot_runs))
        cnt = 1
        param_group = []
        self.acceptance_prob = 0.0

        for itet in range(tot_runs): # number of MCMC steps
            sys.stderr.write("Running chain sample {}\r".format(itet))

            self.sample_from_posterior(x,t)
            with torch.no_grad():
                if store_all and itet>=warmup:
                    if cnt==avoidcorrelation:
                        param_group.append(self.__parameters__(detach = True)) # detach = True will remove the gradient of the parameter removing the cost of the memory of keeping the parameters by 2
                        cnt=1
                        continue
                    cnt+=1
        print("HMC acceptance probability {}".format(self.acceptance_prob/float(tot_runs)*100.))
        return param_group if store_all else self.__parameters__(detach = True)


    def predictive(self,parameters,x):
        with torch.no_grad():
            prediction=0.0
            for idx,params in enumerate(parameters):
                h=x
                for layer,(w,b) in zip(self.Layers,params):
                    layer.set_params(w,b)
                    h=layer(h)
                prediction+=self.softmax(h,dim=1)
            return prediction/float(len(parameters))
