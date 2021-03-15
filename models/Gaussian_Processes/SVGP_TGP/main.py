#-*- coding: utf-8 -*-
# Implementation of SVGP and TGP for regression and classification
# Author: Juan Maroñass

# Python
import argparse
import os
import copy
import sys
sys.path.extend(['../../','../'])

# Standard
import numpy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Torch
import torch
import torch.nn as nn

# Gpytorch
import gpytorch as gpy

# Custom
import config as cg
from dataset import toy_dataset, plot_toy_dataset

from models.models import SVGP, TGP
from models.models.flow import instance_flow
from models.likelihoods import MulticlassCategorical

from pytorchlib import compute_calibration_measures

cg.dtype = torch.float64 # GPs need more precission than NNets :(
torch.set_default_dtype(cg.dtype)

#### ARGUMENT PARSER ####
def parse_args_GP():
    parser = argparse.ArgumentParser(description='Toy example SVGP and TGP measuring accuracy and calibration. Enjoy!')
    parser.add_argument('--epochs'                , type = int, required = True,                                      help = 'optimization epochs'  ) 
    parser.add_argument('--posterior_samples'     , type = int, required = True,                                      help = 'number of samples to draw from the posterior. 1 sample is used for training')
    parser.add_argument('--model'                 , type = str, required = True, choices = ['SVGP', 'TGP', 'ID_TGP'], help = 'model used: SGVP, TGP or input dependent TGP')
    parser.add_argument('--num_inducing'          , type = int, required = True,                                      help = 'number of inducing points')

    parser.add_argument('--plot', type=int,required=True,choices=[0,1],help='plot or not plot')
    args=parser.parse_args()
    return args

#### Function to create a block of SAL flows ####
# This initialization makes the flow a linear function

def SAL(num_blocks, input_dep_specs):

    input_dependent                   = input_dep_specs['input_dependent']
    input_dim, input_dependent_config = None,None

    if input_dependent:
        input_dim              = input_dep_specs['input_dim']
        input_dependent_config = input_dep_specs['input_dependent_config']

    block_array = []
    for nb in range(num_blocks):

        a_aff,b_aff = 1.0,0.0
        a_sal,b_sal = 0.0,1.0

        init_affine = {'init_a':a_aff, 'init_b': b_aff, 'set_restrictions': False}
        init_sinh_arcsinh = {
                              'init_a': a_sal, 'init_b': b_sal,   'add_init_f0': False, 'set_restrictions': False, 
                              'input_dependent': input_dependent, 'input_dim': input_dim, 'input_dependent_config' : input_dependent_config
                            }
        block = [ ('sinh_arcsinh',init_sinh_arcsinh), ('affine',init_affine)  ] 
        block_array.extend(block) 

    return block_array


# Parsing Arguments
args = parse_args_GP()

#### LOAD DATA ####
N_tr = 4000 # number of training samples
N_te = 200  # number of test samples 
MB = N_tr # minibatch. In this case is the same as I am not implementing a dataset interface.
Tr,Te = toy_dataset(N_tr = N_tr, N_te = N_te)
X_tr, T_tr  = Tr
X_te, T_te  = Te

X_tr = X_tr.to(cg.device).to(cg.dtype)
T_tr = T_tr.to(cg.device).view(-1,1)   # The GP models expect two dimensions 
X_te = X_te.to(cg.device).to(cg.dtype)
T_te = T_te.to(cg.device).view(-1,1)

#### Config Variables ####
if args.plot:
    plot_toy_dataset(Tr,Te)

#### Instance the Model ####
num_classes = 4

## initialize inducing points
kmeans = KMeans( n_clusters = args.num_inducing, init = 'k-means++', n_init = 10, random_state = 0).fit(X_tr.to('cpu').numpy())
init_Z = torch.tensor(kmeans.cluster_centers_,dtype=cg.dtype).to(cg.device)

## initialize the kernel to be used
RBF = gpy.kernels.RBFKernel(ard_num_dims = 2 , batch_shape = torch.Size([num_classes]))
RBF.raw_lengthscale.data = gpy.utils.transforms.inv_softplus(torch.ones(num_classes,1,RBF.raw_lengthscale.size(-1),dtype = cg.dtype)*2.0)

Kernel =  gpy.kernels.ScaleKernel(RBF, batch_shape = torch.Size([num_classes]))
Kernel.raw_outputscale.data = gpy.utils.transforms.inv_softplus(torch.ones(num_classes,dtype = cg.dtype)*2.0)

## initialize the mean function to be used
mean = gpy.means.ZeroMean()

## Get likelihood function (use Bernoulli for Binary classification)
likelihood = MulticlassCategorical( num_classes = num_classes )

## Create the flow composition for the TGP
if args.model == 'TGP':

    input_dep_specs = { 'input_dependent' : False }
    flow = SAL( 3, input_dep_specs)

    flow = instance_flow(flow).to(cg.dtype)

    flow = [copy.deepcopy(flow) for i in range(4)] # we create one flow per output, i.e per class
                                                   # need the copy.deepcopy so that each class has its own instance

## Create flow composition for input dependent TGP
elif args.model == 'ID_TGP':

    ## Parameterization of the Bayesian Neural Network used to parameterize the input dependency
    input_dependent_config = {
                                'batch_norm'        : 0,
                                'dropout'           : 0.25,   # note that a value different from 0.0 is needed to use MC dropout inference
                                'hidden_dim'        : 10,     # neuron per layers
                                'hidden_activation' : 'tanh', # activation function. Usually tanh produces a more stable optimization
                                'num_hidden_layers' : 2,      # number of layers
                             }

    input_dep_specs = { 
                        'input_dependent'        : True, 
                        'input_dim'              : 2   ,
                        'input_dependent_config' : input_dependent_config,
                      }

    flow = SAL( 3, input_dep_specs)

    flow = instance_flow(flow).to(cg.device)

    ## now initialize the Neural Network flow parameters that make the flow be the identity mapping. As the SAL flow is initialized to the identity, we just make the nnet parameters
    ## approximate the flow parameters for each of the inputs in the training dataset

    opt = torch.optim.SGD(flow.parameters(), lr = 0.1, momentum = 0.9)
    for _ in range(500):

        loss = flow.forward_initializer(X_tr)
        loss.backward()
        opt.step()
        opt.zero_grad()

        print("Initializer, epoch {} loss {}".format( _ , loss.data), end = "\r")

    flow.turn_off_initializer_parameters() # this just remove the non input dependent parameters that where used for initializing

    flow = [copy.deepcopy(flow) for i in range(4)] # we create one flow per output, i.e per class
                                                   # need the copy.deepcopy so that each class has its own instance

## Instance the model
if args.model in ['TGP', 'ID_TGP']:

    model = TGP(
                    model_specs        = [mean, Kernel] ,
                    init_Z             = init_Z.clone() , 
                    N                  = N_tr           ,
                    likelihood         = likelihood     ,
                    num_outputs        = num_classes    ,
                    is_whiten          = True           ,
                    Z_is_shared        = False          ,
                    flow_specs         = flow           ,
                    be_fully_bayesian  = False          , # This only cares during inference and we change it when evaluating metrics. This is because dropout during training is the same for bayesian 
                                                          # and non-bayesian versions
               )
else:

    model = SVGP(
                    model_specs  = [mean, Kernel] ,
                    init_Z       = init_Z.clone() ,
                    N            = N_tr           ,
                    likelihood   = likelihood     ,
                    num_outputs  = num_classes    ,
                    is_whiten    = True           ,
                    Z_is_shared  = False          ,
                )

model.to(cg.device)

## ===============
## Train the model

## Note we dont have a train method in the class but rather make the loop explicitely
lr = 0.01 if args.model == 'ID_TGP' else 0.1

## add weight decay to Nnet parameters
param_list    = []
id_param_list = []
for n,p in model.named_parameters():
    if 'NNets' in n:
        id_param_list.append(p)
    else:
        param_list.append(p)

param_group    = {'params' : param_list   , 'lr' : lr}
id_param_group = {'params' : id_param_list, 'lr' : lr, 'weight_decay' : 1e-5}
optimizer = torch.optim.Adam([param_group, id_param_group])

## train
for e in range(args.epochs):

    ## compute loss
    loss, ELL, KLD = model.ELBO(X_tr, T_tr) # we are not using minibatching here

    ## negate loss (the ELBO is maximized, hence we minimize the -ELBO
    loss *= -1

    ## back prop
    loss.backward()

    ## optimizer update parameters
    optimizer.step()

    ## optimizer reset
    optimizer.zero_grad()

    ##  Some print
    print("EPOCH: {} \t ELBO {:.3f} ELL {:.3f} KLD {:.3f} ".format(e, -1*loss.item(), ELL.item(), KLD.item() ), end = '\r')

print("\n")

## =====================
## Make some predictions

## Evaluate the moments from the predictive distribution
m1_tr, m2, mean_q_f, cov_q_f = model.predictive_distribution( X = X_tr, diagonal = True, S_MC_NNet = args.posterior_samples) # posterior samples is used to integrate out the Bayesian warping functions. 
                                                                                                                             # The number of monte carlo Samples used to integrate out likelihoods that cant
                                                                                                                             # be integrated with quadrature is given by config.quad_points. You can modify 
                                                                                                                             # the e.g MulticlassCategoricalLikelihood to receive as argument this parameter

m1_te, _, _ , _              = model.predictive_distribution( X = X_te, diagonal = True, S_MC_NNet = args.posterior_samples) 

# for classification we use the first moment (second moment is nonsense): \int p(y| Link(f)) q(f) df = 1/S sum_s p(y | Link(f_s)) ; f_s ~ q(f)

# Some evaluation Metrics
T_tr = T_tr.view(-1)
T_te = T_te.view(-1)

ECEtrain , _ , _ , _ = compute_calibration_measures( m1_tr.float(), T_tr, apply_softmax = False, bins = 15 )
ECEtest  , _ , _ , _ = compute_calibration_measures( m1_te.float(), T_te, apply_softmax = False, bins = 15)

ACCtrain = (float((m1_tr.argmax(1) == T_tr).sum())*100.)/float(T_tr.size(0))
ACCtest  = (float((m1_te.argmax(1)  == T_te).sum())*100.)/float(T_te.size(0))

m = args.model if args.model != 'ID_TGP' else 'Point Estimate TGP'
print("{} \t train acc {:.3f} \t test acc {:.3f}".format(m,ACCtrain,ACCtest))
print("ECE train {:.3f} ECE test {:.3f}".format(ECEtrain,ECEtest))

## now compute bayesian predictions for input dependent
if args.model == 'ID_TGP':
    model.be_fully_bayesian(True)
    m1_tr, m2, mean_q_f, cov_q_f = model.predictive_distribution( X = X_tr, diagonal = True, S_MC_NNet = args.posterior_samples) 
    m1_te, _, _ , _              = model.predictive_distribution( X = X_te, diagonal = True, S_MC_NNet = args.posterior_samples) 

    ECEtrain , _ , _ , _ = compute_calibration_measures( m1_tr.float(), T_tr, apply_softmax = False, bins=15 )
    ECEtest  , _ , _ , _ = compute_calibration_measures( m1_te.float(), T_te, apply_softmax = False, bins=15)
    ACCtrain = (float((m1_tr.argmax(1) == T_tr).sum())*100.)/float(T_tr.size(0))
    ACCtest  = (float((m1_te.argmax(1)  == T_te).sum())*100.)/float(T_te.size(0))

    print("Bayesian TGP \t train acc {:.3f} \t test acc {:.3f}".format(ACCtrain,ACCtest))
    print("ECE train {:.3f} ECE test {:.3f}".format(ECEtrain,ECEtest))

if args.plot:

    cg.device      = 'cpu'
    likelihood.SMC = 100 # to remove noise from the plot
    model          = model.to(cg.device) # change otherwise it runs out of memory in my computer (both CPU and GPU)

    # ===========================================
    # Plot the warping functions learned
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax_list = [ax1, ax2, ax3, ax4]

    f0_range = torch.linspace(-5,5,100).to(cg.device).view(-1,1,1) # to allow broadcasting

    idx = numpy.random.randint(0,N_tr, 4) # get three random training points
    X_range  = X_tr[idx, :].to(cg.device)
    X_range  = X_range.unsqueeze(0).repeat(50,1,1) # dim 0 is to broadcast to f0 and second is to compute the mean and std of the bayesian warping function

    for i,(ax,G) in enumerate(zip(ax_list, model.G_matrix)):
        fk_range = G(f0_range, X_range).detach().cpu()

        fk_mean = fk_range.mean(1).t()
        fk_std  = fk_range.std(1).t()

        for m,s in zip(fk_mean,fk_std):
            ax.plot(f0_range.squeeze().cpu().detach(), m)
            ax.fill_between(f0_range.squeeze().cpu(), m-s, m+s , alpha = 0.2)

        ax.set_title('Class {}'.format(i))

    plt.show(block = False)
    plt.pause(0.01)

    # ===========================================
    # Plot Decision Thresholds learn by the model
   
    model.be_fully_bayesian(False) # change to non-bayesian otherwise I run out of memory

    # Define the grid where we plot
    n_p       = 1000
    vx        = numpy.linspace(-5,4,n_p)
    vy        = numpy.linspace(-5,4,n_p)
    mesh      = numpy.array(numpy.meshgrid(vx, vy))
    data_feat = mesh.T.reshape(-1, 2)

    # forward through the model
    data_feat=torch.from_numpy(data_feat).to(cg.dtype).to(cg.device)
    with torch.no_grad():
        probs, _, _, _    = model.predictive_distribution( data_feat, diagonal = True, S_MC_NNet = args.posterior_samples )
        probs             = probs.cpu().detach()
        max_conf,max_labl = torch.max( probs, dim=1 )

    X,Y=mesh[0],mesh[1]

    max_conf = max_conf.reshape(n_p,n_p).T
    max_labl = max_labl.reshape(n_p,n_p).T
    aux      = numpy.zeros((n_p,n_p),numpy.float32)

    cmap = [plt.cm.get_cmap("Reds"),plt.cm.get_cmap("Greens"),plt.cm.get_cmap("Blues"),plt.cm.get_cmap("Greys")]

    color_list_tr=['r','g','b','k']
    color_list_te=['r','g','b','k']

    markers    = ['*','*','*','*']
    markers_te = ['o','o','o','o']

    fig,ax = plt.subplots(figsize=(25,35))

    for ctr,cte,i,c,marker, marker_te in zip(color_list_tr,color_list_te,range(4),cmap,markers, markers_te):

        idx_tr = T_tr == i
        idx_te = T_te == i
        xtr = X_tr[idx_tr,:]
        xte = X_te[idx_te,:]

        ## Plot training and test samples 
        x1,x2 = xtr[:,0].cpu().numpy(),xtr[:,1].cpu().numpy()
        ax.plot( x1, x2, marker, color=ctr, markersize = 5, alpha = 1.0)

        x1,x2 = xte[:,0].cpu().numpy(),xte[:,1].cpu().numpy()
        ax.plot(x1, x2, marker_te, color=cte, markersize = 5, alpha = 1.0)

        
        ## Plot training and test samples 
        x1,x2=xtr[:,0].cpu().numpy(),xtr[:,1].cpu().numpy()
        ax.plot(x1,x2,marker, color = cte, markersize = 5, alpha = 1.0)

        ## Plot the winning decision threshold 
        idx_row, idx_cols     = numpy.where(max_labl==i)
        aux[idx_row,idx_cols] = max_conf[idx_row,idx_cols]

        idx_row, idx_cols      = numpy.where(max_labl!=i)
        aux[idx_row, idx_cols] = numpy.nan

        dib=ax.contourf(X,Y,aux,cmap=c,alpha=0.5,levels=[0,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.92,0.94,0.96,0.98,1.0])

        if i==3:
            plt.xlabel('x1',fontsize=70)
            plt.ylabel('x2',fontsize=70)
            plt.xticks([-5,-4,-3,-2,-1,0,1,2,3,4], fontsize = 60) 
            plt.yticks([-5,-4,-3,-2,-1,0,1,2,3,4], fontsize = 60) 
            
                    
plt.show()	



