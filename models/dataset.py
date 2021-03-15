# torch
import torch

# custom 
import config
from config import reset_seed

# Standard
import numpy
import matplotlib.pyplot as plt

###########################
####### TOY DATASET #######

def toy_dataset( N_tr: int, N_te : int ) -> list:
    ''' Toy Dataset 
            Args:
                    N  (int) :->: Number of Samples. If it is not divisible by 4, it will be rounded to be
    '''
    reset_seed(1)

    per_class = int(N_te/4.)

    ''' test set '''
    x1=torch.randn(per_class,2)*1+0.5
    x2=torch.randn(per_class,2)*0.5+torch.from_numpy(numpy.array([0.5,-2])).float()
    x3=torch.randn(per_class,2)*0.3-0.5
    x4=torch.randn(per_class,2)*0.8+torch.from_numpy(numpy.array([-1.0,-2])).float()

    t1=torch.zeros(per_class,)
    t2=torch.ones(per_class,)
    t3=torch.ones(per_class,)+1
    t4=torch.ones(per_class,)+2

    idx=numpy.random.permutation(per_class*4)
    x_test=torch.cat((x1,x2,x3,x4))[idx].float()
    t_test=torch.cat((t1,t2,t3,t4))[idx].long()

    ''' train set '''
    per_class = int(N_tr/4.)
    
    #sample samples per class
    x1=torch.randn(per_class,2)*1.0+0.5
    x2=torch.randn(per_class,2)*0.5+torch.from_numpy(numpy.array([0.5,-2])).float()
    x3=torch.randn(per_class,2)*0.3-0.5
    x4=torch.randn(per_class,2)*0.8+torch.from_numpy(numpy.array([-1.0,-2])).float()

    t1=torch.zeros(per_class,)
    t2=torch.ones(per_class,)
    t3=torch.ones(per_class,)+1
    t4=torch.ones(per_class,)+2

    idx=numpy.random.permutation(per_class*4)
    x=torch.cat((x1,x2,x3,x4))[idx].float()
    t=torch.cat((t1,t2,t3,t4))[idx].long()

    return [x,t],[x_test,t_test]

colors = ['b','g','r','k']
colors_test = ['cyan', 'lightgreen', 'orange', 'gray']

def plot_toy_dataset(X_tr: list , X_te : list ) -> None:

    X,T = X_tr
    X_te, T_te = X_te

    X = X.detach().numpy()
    T = T.detach().numpy()
    X_te = X_te.detach().numpy()
    T_te = T_te.detach().numpy()


    N_labels = numpy.unique(T)
    assert len(N_labels) == len(numpy.unique(T_te)), "Getting different number of classes"
    
    for l in N_labels:
        idx_tr = T == l
        idx_te  = T_te == l
        plt.plot(X[idx_tr,0],X[idx_tr,1],'*'+colors[l])
        plt.plot(X_te[idx_te,0],X_te[idx_te,1],'o',color=colors_test[l])

    plt.show(block = False)
    plt.pause(0.01)


