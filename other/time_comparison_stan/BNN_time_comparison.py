## Author: Juan Maroñas Molano
## Interface to launch several stan programs implementing a Bayesian Categorical regresion with softmax link function.
## The script allows to compile using openmpi. However, stan code only uses reduce_sum for parallelization (no map). So this
## basically means that compiling with open_mpi makes no difference. I didnt know this feature previous to code this example
## an this information is not provided in stan doc, so got it by chance through a discussion in the stan forum
## which is provided here: https://discourse.mc-stan.org/t/measuring-and-comparing-computational-performance-in-stan-with-different-compilation-alternatives-using-reduce-sum-does-not-bring-any-advantage/21340/18

import numpy
from cmdstanpy import CmdStanModel
import cmdstanpy
import os
import argparse

## =================================================================================== ##
## ======================= Parsing Experiment Arguments ============================== ##
## =================================================================================== ##
parser = argparse.ArgumentParser( description = 'Performance Comparison in Stan' )

## == Model == ##
parser.add_argument('--model_type', required = True, help = 'Which model to run',
                    choices = [ 
                                'no_partial_sum_categorical_GLM',
                                'no_partial_sum_categorical_lupmf_column_indexing',
                                'no_partial_sum_categorical_lupmf_column_indexing_with_transposition',
                                'no_partial_sum_categorical_lupmf_row_indexing',
                                'partial_sum_SLICED_ARGS_logit_SHARED_ARGS_y_row_indexing_and_transposition',
                                'partial_sum_SLICED_ARGS_logit_SHARED_ARGS_y',
                                'partial_sum_SLICED_ARGS_y_SHARED_ARGS_logit_column_indexing',
                                'partial_sum_SLICED_ARGS_y_SHARED_ARGS_logit_column_indexing_with_transposition',
                                'partial_sum_SLICED_ARGS_y_SHARED_ARGS_logit_row_indexing',
                                'partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_Categorical_GLM_columns_indexing',
                                'partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_Categorical_GLM_row_indexing',
                                'partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_column_indexing',
                                'partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_row_indexing',
                              ]
                    )

# == Compilation options == #
parser.add_argument('--compile_gpu',     action='store_true', default = False, help = 'Compile in gpu')
parser.add_argument('--compile_openmpi', action='store_true', default = False, help = 'Compile with openmpi')
parser.add_argument('--stan_threads', required = False, default = None, type = int, help = 'Number of threads used with open mpi')

# == Config Parsing == #
args = parser.parse_args()


## =================================================================================== ##
## ===============    Configuration selected to be launched   ======================== ##
## =================================================================================== ##
model_type = args.model_type

## =================================================================================== ##
## ======================    Code compilation flags    =============================== ##
## =================================================================================== ##

if 'no_partial_sum' in model_type:
    assert args.compile_openmpi == False, "You have set model to be compiled with openmpi but the current stan model does not work with a partial_sum reduction, hence multithreading will not be performed using stan threads."

    assert args.stan_threads == None, "you set the model to use stan threads but the current model has no partial_sum reduction, hence multithreading cannot be performed"

## Choose model compilation
compile_gpu     = args.compile_gpu
compile_openmpi = args.compile_openmpi
stan_threads    = args.stan_threads

cpp_options = {}

if args.stan_threads is not None:
    cpp_options['STAN_THREADS'] = stan_threads

if compile_openmpi:
    cpp_options['STAN_MPI']     = 'TRUE'

if compile_gpu:
    cpp_options['STAN_OPENCL'] = 'TRUE'

## Compile the program
sm = CmdStanModel(
                  stan_file   = './stan_files/BNN_{}.stan'.format(model_type), 
                  cpp_options = cpp_options
                 )
 

## =================================================================================== ##
## ==========================      Data Loading        =============================== ##
## =================================================================================== ##

## Generate some random data
N = 45000
C = 10

x = numpy.random.randn(N,C)
y = numpy.random.randint(C, size = (N,))

y += 1


## ==================================
## Prepare input data to stan program 
N = x.shape[0]
C = x.shape[1]

if model_type == 'no_partial_sum_categorical_lupmf_column_indexing':
    x = x.transpose()

if model_type == 'partial_sum_SLICED_ARGS_logit_SHARED_ARGS_y':
    x = x.transpose()

if model_type == 'partial_sum_SLICED_ARGS_y_SHARED_ARGS_logit_column_indexing':
    x = x.transpose()

if model_type == 'partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_Categorical_GLM_columns_indexing':
    x = x.transpose()

if model_type == 'partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_column_indexing':
    x = x.transpose()

## prior definition
mu_w   = 0.0
std_w  = 1.0

BNN_data = {
                'N'     : N,
                'C'     : C,
                'x'     : x,
                'y'     : y,
                'mu_w'  : mu_w,
                'std_w' : std_w,
            }

## =================================================================================== ##
## ==================      Directory to save results      ============================ ##
## =================================================================================== ##

with_gpu = 'cpu'
if compile_gpu:
    with_gpu = 'cuda'

with_openmpi = 'NO_threads_1'

if stan_threads is not None:
    with_openmpi = 'NO_threads_{}'.format(stan_threads)

if compile_openmpi:
    with_openmpi = 'YES_threads_{}'.format(stan_threads)

output_dir = os.path.join('./results', with_gpu, with_openmpi, model_type)


## =================================================================================== ##
## ==========================      Run Inference          ============================ ##
## =================================================================================== ##

## ==========================
## Regarding Chain Generation

iter_sampling = 1      # number of samples to draw after warm up
chains        = 1      # number of chains sampled in parallel
iter_warmup   = 1000   # number of iterations of the warm up stage
thin          = 1      # chain thinning. Can improve ESS

## ====================
## NUTS sampler
max_treedepth = 10

## Step size adaptation
delta = 0.8  # acceptance rate probabilty 
# Below parameters are not available through the cmdstanpy interface 
# gamma = 0.05 # from nesterov algorithm. Stan recommends default
# kappa = 0.75 # from nesterov algorithm. Stan recommends default  
# t_0   = 10   # from nesterov algorithm. Stan recommends default


## Adaptation stage
adapt_engaged       = False  # If false no adaptation is done. We just want to measure sampling time
adapt_init_phase    = 75 # correspond to adaptation step I   in Stan reference manual. Specifies width in samples
adapt_metric_window = 25 # correspond to adaptation step II  in Stan reference manual. Specifies width in samples
adapt_step_size     = 50 # correspond to adaptation step III in Stan reference manual. Specifies width in samples

## ====================
## Specification for kinetic energy
metric_M = 'diag_e' # the shape of the correlation matrix M. This is very important, check 2014 MLSS talk from Betancourt on youtube
                     # options diag_e. dense_e will handle correlations 

if not adapt_engaged:
    adapt_init_phase    = None
    adapt_metric_window = None 
    adapt_step_size     = None
    delta               = None
    iter_warmup         = None


## ===============
## Run the sampler

fit_sampler  = sm.sample(
                         # data passed to stan
                         data          = BNN_data, 
    
                         # ===
                         # specification about the chain generated
                         iter_sampling     = iter_sampling, 
                         chains            = chains, 
                         threads_per_chain = stan_threads, # number of threads per chain when STAN_THREADS are activated.
                                                           # this threads are used to parallelize the reduce_sum computations        
    
                         iter_warmup   = iter_warmup, 
                         save_warmup   = True,  # warm up samples are kept or not
    
                         seed          = 1,
                    
                         # ===
                         # optimization of step size
                         adapt_delta = delta,
                         # gamma = gamma,
                         # kappa = kappa,
                         # t_0   = t_0,
                    
                         # ===
                         # adaptation stage
                         adapt_engaged       = adapt_engaged,           
                         adapt_init_phase    = adapt_init_phase,
                         adapt_metric_window = adapt_metric_window,
                         adapt_step_size     = adapt_step_size,

                         # ===
                         # Kinetic energy
                         metric = metric_M,
    
                         # ===
                         # Config things
                         output_dir    = output_dir,
                         save_diagnostics = True
                        )





