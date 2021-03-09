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
import config
device = config.device
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
N = 400 # number of training samples
MB = N # minibatch. In this case is the same as I am not implementing a dataset interface.
Tr,Te = toy_dataset(N)
X_tr, T_tr  = Tr
X_te, T_te  = Te


#### Config Variables ####
if visualize:
    plot_toy_dataset(Tr,Te)

if args.Local_rep == 0:
    net = BNN_VI(neu_layers,num_layers,2,4,args.prior_mean,args.prior_var,dataset_size = N)
else:
    net = BNN_VILR(neu_layers,num_layers,2,4,args.prior_mean,args.prior_var,batch = N, dataset_size = N)
net.to(device)

#######"TRAIN/INFERENCE" MODEL ######

def scheduler(lr,eps):
    if eps<-1100:
            lr=lr/10.
    return lr

# Train the model
with torch.autograd.set_detect_anomaly(True):
    net.train(X_tr,T_tr,scheduler=scheduler,epochs=args.epochs,lr=0.1,warm_up=10, MC_samples = 10)

# Inference in the model
prediction_train=net.predictive(X_tr,predictive_samples)
prediction_test=net.predictive(X_te,predictive_samples)

# Some evaluation Metrics
ECEtrain,_,_,_=compute_calibration_measures(prediction_train,T_tr,apply_softmax=False,bins=15)
ECEtest,_,_,_=compute_calibration_measures(prediction_test,T_te,apply_softmax=False,bins=15)
ACCtrain=(float((prediction_train.argmax(1)==T_tr).sum())*100.)/float(T_tr.size(0))
ACCtest=(float((prediction_test.argmax(1)==T_te).sum())*100.)/float(T_te.size(0))

print("MAP net \t train acc {} \t test acc {}".format(ACCtrain,ACCtest))
print("ECE train {} ECE test {}".format(ECEtrain,ECEtest))

# Plot Decision Thresholds learn by the model
if visualize:
    # Define the grid where we plot
    vx=numpy.linspace(-5,4,1000)
    vy=numpy.linspace(-5,4,1000)
    data_feat=numpy.zeros((1000000,2),numpy.float32)

    # this can be done much more efficient for sure
    for x,px in enumerate(vx):
        for y,py in enumerate(vy):
            data_feat[x*1000+y]=numpy.array([px,py])

    # forward through the model
    data_feat=torch.from_numpy(data_feat)	
    with torch.no_grad():
        logits=net.predictive(data_feat,args.predictive_samples)
        max_conf,max_target=torch.max(logits,dim=1)


    conf=numpy.zeros((1000000),numpy.float32)
    labl=numpy.zeros((1000000),numpy.float32)
    data_feat=0
    conf[:]=numpy.nan
    labl[:]=numpy.nan
    max_conf,max_target=max_conf.detach(),max_target.detach()
    for x,px in enumerate(vx):
        for y,py in enumerate(vy):
            conf[x*1000+y]=max_conf[x*1000+y]
            labl[x*1000+y]=max_target[x*1000+y]
    
    X,Y=numpy.meshgrid(vx,vy)

    cmap = [plt.cm.get_cmap("Reds"),plt.cm.get_cmap("Greens"),plt.cm.get_cmap("Blues"),plt.cm.get_cmap("Greys")]

    color_list_tr=['*r','*g','*b','*k']
    color_list_te=['orange','lightgreen','cyan','gray']
    markers=['d','*','P','v']
    fig,ax = plt.subplots(figsize=(25,35))

    for ctr,cte,i,c,marker in zip(color_list_tr,color_list_te,range(4),cmap,markers):

        idx_tr = T_tr == i
        idx_te = T_te == i
        xtr = X_tr[idx_tr,:]
        xte = X_te[idx_te,:]
        
        aux=numpy.zeros((1000000),numpy.float32)
        x1,x2=xtr[:,0].numpy(),xtr[:,1].numpy()

        plt.plot(x1,x2,marker,color=cte,markersize=20,alpha=0.5)

        index=numpy.where(labl==i)[0]
        aux[index]=conf[index]
        index_neg=numpy.where(labl!=i)[0]
        aux[index_neg]=numpy.nan
        aux=aux.reshape(1000,1000,order='F')
        dib=ax.contourf(X,Y,aux,cmap=c,alpha=0.5,levels=[0,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.92,0.94,0.96,0.98,1.0])

        if i==3:
            plt.xlabel('x1',fontsize=70)
            plt.ylabel('x2',fontsize=70)
            plt.xticks([-5,-4,-3,-2,-1,0,1,2,3,4], fontsize = 60) 
            plt.yticks([-5,-4,-3,-2,-1,0,1,2,3,4], fontsize = 60) 
            
                    
plt.show()	
