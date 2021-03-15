## Neural Network with point estimation obtained via Maximum a posterio
## Author: Juan Maroñas Molano

# Torch
import torch
import torch.nn as nn

# Python
import sys
sys.path.extend(['../../../'])
import math

# Custom
import config as cg


class FC(nn.Module):
    def __init__(self,indim,outdim,activation):
        super(FC, self).__init__()
        self.w = nn.Parameter(torch.rand((indim,outdim)))
        self.b = nn.Parameter(torch.rand((outdim)))
        self.activation = nn.functional.relu if activation=='relu' else None

        nn.init.uniform_(self.b,-0.1,0.1)
        nn.init.kaiming_uniform_(self.w, a = math.sqrt(5))

    def forward(self,x):
        return self.activation(torch.mm(x,self.w)+self.b) if self.activation != None else torch.mm(x,self.w)+self.b


class MAP_NN(nn.Module):
    def __init__(self,neur,number_layers,in_dim,out_dim,prior_mean,prior_var):
        super(MAP_NN, self).__init__()

        if number_layers == 0:
            Lin=FC(in_dim,out_dim,'linear')
            self.Layers=nn.ModuleList([Lin])
        else:
            Lin=FC(in_dim,neur,'relu')
            self.Layers=nn.ModuleList([Lin])
            for l in range(number_layers-1):
                    self.Layers.append(FC(neur,neur,'relu'))
            self.Layers.append(FC(neur,out_dim,'linear'))

        self.ce=nn.functional.cross_entropy
        self.softmax=nn.functional.softmax
        self.prior_mean=torch.tensor(prior_mean).to(cg.device)
        self.prior_var=torch.tensor(prior_var).to(cg.device)
            
    def forward(self,x):
        for l in self.Layers:
            x=l(x)

        return x

    def return_log_prior(self):
        acc=0
        for l in self.Layers:
            acc += ( -0.5*math.log(2*math.pi) -0.5*torch.log(self.prior_var) - 1./(2.*self.prior_var)*((l.w-self.prior_mean)**2)).sum()
            acc += ( -0.5*math.log(2*math.pi) -0.5*torch.log(self.prior_var) - 1./(2.*self.prior_var)*((l.b-self.prior_mean)**2)).sum()
        return acc

    def train(self, x, t, epochs, lr, scheduler):

        # please note that we can simply pass the weight decay to the optimizer, however to make explicit what a gaussian prior over the parameters does I included in the cost function to be minimized. 
        # Also there is a difference in the scale magnitude of the gradient when using weight decay vs the actual log prior

        optimizer=torch.optim.SGD(self.parameters(),lr=lr,momentum=0.9) # SGD goes better for point estimate models

        for e in range(epochs):

            lr_=scheduler(lr,e)

            for idx in range(len(optimizer.param_groups)):
                optimizer.param_groups[idx]['lr'] = lr_

            logit = self.forward(x)
            loss  = self.ce(logit,t, reduction = 'sum' ) + -1 * self.return_log_prior() # -1 comes from the fact that we minimize the - log p(w|X) = - log p(t|x) - log p(w)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("On epoch {} Loss {:.5f}".format(e,loss.data))				
            
