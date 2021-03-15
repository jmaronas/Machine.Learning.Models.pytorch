#-*- coding: utf-8 -*-
# flow.py : file containing flow base class
# Author: Juan Maroñas

# Python
import sys
sys.path.extend(['../../../../'])

# Torch
import torch
import torch.nn as nn
from torch.nn.functional import softplus

# Gpytorch
import gpytorch
from gpytorch.utils.transforms import inv_softplus

## Pytorch lib
from pytorchlib import apply_linear 

# custom
import config as cg

def instance_flow(flow_list, is_composite = True):
    ''' From these flows only Box-Cox, sinh-arcsinh and affine return to the identity '''
    FL = []
    for flow_name in flow_list:

        flow_name,init_values = flow_name

        if flow_name == 'affine':
            fl = AffineFlow(**init_values)

        elif flow_name == 'sinh_arcsinh':
            fl = Sinh_ArcsinhFlow(**init_values)

        elif flow_name == 'identity':
            fl = IdentityFlow()

        else:
            raise ValueError("Unkown flow identifier {}".format(flow_name))

        FL.append(fl)

    if is_composite:
        return CompositeFlow(FL)
    return FL

class Flow(nn.Module):
    """ General Flow Class. 
        All flows should inherit and overwrite this method
    """
    def __init__(self) -> None:
        super(Flow, self).__init__()

    def forward(self, f0:torch.tensor, X:torch.tensor = None) -> torch.tensor:
        raise NotImplementedError("Not Implemented")

    def turn_off_initializer_parameters(self):
        raise NotImplementedError("Not Implemented")

    def forward_initializer(self,X):
        # just return 0 if it is not needed
        raise NotImplementedError("Not Implemented")
    

class CompositeFlow(Flow):
    def __init__(self, flow_arr: list) -> None:
        """
            Args:
                flow_arr: is an array of flows. The first element is the first flow applied.
        """
        super(CompositeFlow, self).__init__()
        self.flow_arr   = nn.ModuleList(flow_arr)
        self.is_inp_dep = [f.input_dependent for f in flow_arr] ## keeps if each of the flows is input dependent or not. In this way we can combine input and non input dependent flows and the adequate
                                                                ## initiaizerss

    def forward(self, f: torch.tensor, X:torch.tensor = None) -> torch.tensor :
        for flow in self.flow_arr:
            f = flow.forward(f,X)
        return f

    def forward_initializer(self, X : torch.tensor):
        loss = 0.0
        for flow in self.flow_arr:
            loss += flow.forward_initializer(X)
        return loss

    @property
    def input_dependent(self):
        pass

    @input_dependent.setter
    def input_dependent(self,value):
        for is_inp_dep, flow in zip(self.is_inp_dep, self.flow_arr):
            if is_inp_dep:
                flow.input_dependent = value
                
    def turn_off_initializer_parameters(self):
        for flow in self.flow_arr:
            flow.turn_off_initializer_parameters()      

class IdentityFlow(Flow):
    """ Identity Flow
           fk = f0
    """
    def __init__(self) -> None:
        super(IdentityFlow,self).__init__()
        self.input_dependent  = False

    def forward(self, f0 : torch.tensor, X : torch.tensor = None) -> torch.tensor:
        return f0

class AffineFlow(Flow):
    def __init__(self, init_a: float , init_b: float, set_restrictions : bool, input_dependent : bool = False, input_dim : float = -1, input_dependent_config: dict = {}) -> None:
        ''' Affine Flow
            fk = a*f0+b 
            * recovers the identity for a = 1 b = 0
            * a has to be strictly possitive to ensure invertibility if this flow is used in a linear
            combination, i.e with the step flow
            Args:
                a                (float) :->: Initial value for the slope
                b                (float) :->: Initial value for the bias
                set_restrictions (bool)  :->: If true then a >= 0 using  a = softplus(a)
        '''
        super(AffineFlow,self).__init__()
        self.a = nn.Parameter(torch.tensor(init_a,dtype=cg.dtype))
        self.b = nn.Parameter(torch.tensor(init_b,dtype=cg.dtype))

        self.set_restrictions = set_restrictions
        self.input_dependent  = input_dependent

    def forward(self, f0:torch.tensor, X:torch.tensor = None) -> torch.tensor:

        if self.input_dependent: 
            raise NotImplementedError()

        else:
            a = self.a
            if self.set_restrictions:
                a = softplus(a)
            b = self.b
            return a*f0 + b

    def forward_initializer(self,X):

        if self.input_dependent:
            raise NotImplementedError()
        else:
            return 0.0

    def turn_off_initializer_parameters(self):
        pass


class Sinh_ArcsinhFlow(Flow):
    def __init__(self,init_a:float, init_b:float, add_init_f0:bool, set_restrictions:bool, input_dependent:bool = False, input_dim: float = -1, input_dependent_config: dict = {}) -> None:
        ''' SinhArcsinh Flow
          fk = sinh( b*arcsinh(f) - a)
          * b has to be strictkly possitive when used in a linear combination so that function is invertible.
          * Recovers the identity function
            
          Args:
                 init_a           (float) :->: initial value for a. Only used if input_dependent = False. Also used by the initializer if input_dependent = True so 
                                               that NNets parameters are matched to take this value.
                 init_b           (float) :->: initial value for b. Only used if input_dependent = False. Also used by the initializer if input_dependent = True so 
                                               that NNets parameters are matched to take this value.
                 set_restrictions (bool)  :->: if true then b > 0 with b = softplus(b)
                 add_init_f0      (bool)  :->: if true then fk = f0 + sinh( b*arcsinh(f) - a)
                                               If true then set_restrictions = True
                 input_dependent  (bool)  :->: If true the parameters of the flow depend on the input
        '''
        super(Sinh_ArcsinhFlow, self).__init__()

        if input_dependent:

            assert input_dim > 0, "Set input dimension if input_dependent = True"
            BN,DR,H,act,num_H = 0,0.0,input_dim,'relu',1

            if 'batch_norm' in input_dependent_config.keys():
                BN = input_dependent_config['batch_norm']
            if 'dropout'    in input_dependent_config.keys():
                DR = input_dependent_config['dropout']
            if 'hidden_dim' in input_dependent_config.keys():
                H  = input_dependent_config['hidden_dim']
            if 'hidden_activation' in input_dependent_config.keys():
                act  = input_dependent_config['hidden_activation'] 
            if 'num_hidden_layers' in input_dependent_config.keys():
                num_H = input_dependent_config['num_hidden_layers']

            list_a,list_b = [],[]

            # Parameter a
            inp_dim = input_dim
            for _ in range(num_H):
                list_a.append(apply_linear(inp_dim,H, act, shape = None, std = 0.0, drop = DR, bn = BN))
                inp_dim = H
            list_a.append(apply_linear(H, 1, 'linear', shape = None, std = 0.0, drop = 0.0, bn = 0))

            # Parameter b
            inp_dim = input_dim
            for _ in range(num_H):
                list_b.append(apply_linear(inp_dim, H, act, shape = None, std = 0.0, drop = DR, bn = BN))
                inp_dim = H
            list_b.append(apply_linear(H, 1, 'linear', shape = None, std = 0.0, drop = 0.0, bn = 0))

            self.NNets_a = nn.Sequential(*list_a)
            self.NNets_b = nn.Sequential(*list_b)


            # These are only used by the input dependent initializer
            self.a = nn.Parameter(torch.tensor(init_a,dtype=cg.dtype))
            self.b = nn.Parameter(torch.tensor(init_b,dtype=cg.dtype))

            self.parameters_are_turn_off = False
            self.is_using_bn = True if BN == 1 else False

        else:
            self.a = nn.Parameter(torch.tensor(init_a,dtype=cg.dtype))
            self.b = nn.Parameter(torch.tensor(init_b,dtype=cg.dtype))

        if add_init_f0:
            set_restrictions = True

        self.set_restrictions = set_restrictions
        self.add_init_f0      = add_init_f0

        self.input_dependent = input_dependent

    def asinh(self, f: torch.tensor ) -> torch.tensor :
        return torch.log(f+(f**2+1)**(0.5))

    def forward_initializer(self,X):
        if self.input_dependent:
            a = self.NNets_a(X)
            b = self.NNets_b(X)
    
            A = ((a-self.a.detach())**2).mean()
            B = ((b-self.b.detach())**2).mean()

            return A+B 
        else:
            return 0.0

    def turn_off_initializer_parameters(self):
        '''Set self.a and self.b requires grad to False. Those parameters are only used for initialization of the input dependent ones, 
           hence switch them of to not be tracked by the optimizer. They wont be used by the ELBO, but to be bug safe is better to force this call
        '''
        if self.input_dependent:
            if not self.parameters_are_turn_off:
                self.a_untracked = self.a.data.detach() # keep a copy in case we want to check it. We cant reasign self.a with something different to nn.Parameter or None.
                self.b_untracked = self.b.data.detach()
                self.a = None # switch it to torch.tensor
                self.b = None 
            self.parameters_are_turn_off = True

    def forward(self, f0:torch.tensor, X:torch.tensor = None) -> torch.tensor:
        #assert self.is_initialized, "This flow hasnt been initialized. Either set self.is_initialized = False or use an initializer"

        if self.input_dependent:
            assert X != None, "Set X to value"
            assert self.parameters_are_turn_off, "Call the method turn_off_initializer_parameters before using the flow in an optimization loop. Keep yourself bug safe man."
            # Check here for the Deep Model 

            if self.is_using_bn:
                x_shape = X.shape[:-1]
                ld      = X.shape[-1]
                X = X.reshape(numpy.prod(list(x_shape)),ld)

            a = self.NNets_a(X)
            b = self.NNets_b(X)

            if self.is_using_bn:
                ld = a.shape[-1]
                a = a.view(x_shape+(ld,))

                ld = b.shape[-1]
                b = b.view(x_shape+(ld,))


            a = a.squeeze(dim = -1)
            b = b.squeeze(dim = -1)

            if self.set_restrictions:
                b = softplus(b)
            fk = torch.sinh( b * self.asinh(f0) - a )

        else:
            a = self.a
            b = self.b
            if self.set_restrictions:
                b = softplus(b)
            fk = torch.sinh( b * self.asinh(f0) - a )
            
        if self.add_init_f0:
                return fk + f0

        return fk


