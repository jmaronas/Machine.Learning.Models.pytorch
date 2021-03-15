## Neural Network with point estimation obtained via Maximum Likelihood
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
        self.w = nn.Parameter(torch.randn((indim,outdim)))
        self.b = nn.Parameter(torch.randn((outdim)))

        self.activation=nn.functional.relu if activation=='relu' else None

        nn.init.uniform_(self.b,-0.1,0.1)
        nn.init.kaiming_uniform_(self.w, a = math.sqrt(5))

    def forward(self,x):
        return self.activation(torch.mm(x,self.w)+self.b) if self.activation != None else torch.mm(x,self.w)+self.b

class ML_NN(nn.Module):
    def __init__(self,neur,number_layers,input_dim,out_dim):

        super(ML_NN, self).__init__()

        if number_layers == 0:

            Lin=FC(input_dim,out_dim,'linear')
            self.Layers=nn.ModuleList([Lin])

        else:

            Lin=FC(input_dim,neur,'relu')
            self.Layers=nn.ModuleList([Lin])
            for l in range(number_layers-1):
                    self.Layers.append(FC(neur,neur,'relu'))

            self.Layers.append(FC(neur,out_dim,'linear'))

        self.ce=nn.functional.cross_entropy
        self.softmax=nn.functional.softmax

    def forward(self,x):
        for l in self.Layers:
            x=l(x)

        return x

    def train(self,x,t,epochs,lr, scheduler):

        optimizer=torch.optim.SGD(self.parameters(),lr=lr,momentum=0.9)#SGD goes better for point estimate models
        for e in range(epochs):
            
            lr_=scheduler(lr,e)

            for idx in range(len(optimizer.param_groups)):
                optimizer.param_groups[idx]['lr'] = lr_

            logit = self.forward(x)
            loss  = self.ce(logit,t, reduction = 'sum')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("On epoch {} Loss {:.5f}".format(e,loss.data))				
            

