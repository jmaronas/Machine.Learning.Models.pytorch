## Bayesian Neural Network with Mean Field Variational inference with and without Local Reparameterization

## Author: Juan Maroñas Molano
# Standard
import matplotlib.pyplot as plt
import numpy

# torch
import torch
import torch.nn as nn

# Python
import argparse
import os
import sys
sys.path.extend(['../../','../'])

# Custom
import config as cg
from pytorchlib import compute_calibration_measures
from models import  BNN_VILR, BNN_VI
from dataset import toy_dataset, plot_toy_dataset

#### ARGUMENT PARSER ####
def parse_args_VI():
    parser = argparse.ArgumentParser(description='Toy example point estimate Models measuring accuracy and calibration. Enjoy!')
    parser.add_argument('--epochs', type=int,required=True,help='epochs')
    parser.add_argument('--predictive_samples', type=int,required=True,help='number of samples to compute the predictive distribution')
    parser.add_argument('--num_layers', type=int,required=True,help='hidden layers')
    parser.add_argument('--neu_layers', type=int,required=True,help='neurons per hidden layer')
    parser.add_argument('--prior_mean', type=float,required=True,help='prior mean')
    parser.add_argument('--prior_var', type=float,required=True,help='prior variance')
    parser.add_argument('--Local_rep', type=int,required=True,choices=[0,1],help='apply local reparameterization')
    parser.add_argument('--plot', type=int,required=True,choices=[0,1],help='plot or not plot')
    args=parser.parse_args()
    return args

# Parsing Arguments
args = parse_args_VI()
num_layers=args.num_layers
neu_layers=args.neu_layers
predictive_samples=args.predictive_samples
visualize = args.plot

#### LOAD DATA ####
N_tr = 4000 # number of training samples
N_te = 200  # number of test samples
MB = N_tr # minibatch. In this case is the same as I am not implementing a dataset interface.
Tr,Te = toy_dataset(N_tr = N_tr , N_te = N_te)
X_tr, T_tr  = Tr
X_te, T_te  = Te

X_tr = X_tr.to(cg.device)
T_tr = T_tr.to(cg.device)
X_te = X_te.to(cg.device)
T_te = T_te.to(cg.device)

#### Config Variables ####
if visualize:
    plot_toy_dataset(Tr,Te)

if args.Local_rep == 0:
    net = BNN_VI(neu_layers,num_layers,2,4,args.prior_mean,args.prior_var,dataset_size = N_tr, use_batched_computations = 0)
else:
    net = BNN_VILR(neu_layers,num_layers,2,4,args.prior_mean,args.prior_var, batch = MB, dataset_size = N_tr, use_batched_computations = 1)
net.to(cg.device)

#######"TRAIN/INFERENCE" MODEL ######

def scheduler(lr,eps):
    if eps > 500:
        lr = lr/10.

    return lr

# Train the model
with torch.autograd.set_detect_anomaly(True):
    net.train(X_tr,T_tr,scheduler=scheduler,epochs=args.epochs, lr = 0.1, warm_up = 10, MC_samples = 10)

# Inference in the model
prediction_train=net.predictive(X_tr,predictive_samples)
prediction_test=net.predictive(X_te,predictive_samples)

# Some evaluation Metrics
ECEtrain,_,_,_=compute_calibration_measures(prediction_train,T_tr,apply_softmax=False,bins=15)
ECEtest,_,_,_=compute_calibration_measures(prediction_test,T_te,apply_softmax=False,bins=15)
ACCtrain=(float((prediction_train.argmax(1)==T_tr).sum())*100.)/float(T_tr.size(0))
ACCtest=(float((prediction_test.argmax(1)==T_te).sum())*100.)/float(T_te.size(0))

print("VI BNN net \t train acc {:.3f} \t test acc {:.3f}".format(ACCtrain,ACCtest))
print("ECE train {:.3f} ECE test {:.3f}".format(ECEtrain,ECEtest))

# Plot Decision Thresholds learn by the model
if visualize:

    net.use_batched_computations = 0 # change otherwise it runs out of memory in my computer (both CPU and GPU)

    # Define the grid where we plot
    n_p       = 1000
    vx        = numpy.linspace(-5,4,n_p)
    vy        = numpy.linspace(-5,4,n_p)
    mesh      = numpy.array(numpy.meshgrid(vx, vy))
    data_feat = mesh.T.reshape(-1, 2)

    # forward through the model
    data_feat=torch.from_numpy(data_feat).to(cg.dtype).to(cg.device)
    with torch.no_grad():
        logits=net.predictive(data_feat,args.predictive_samples, use_same_samples = True).cpu().detach()
        max_conf,max_labl=torch.max(logits,dim=1)

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

