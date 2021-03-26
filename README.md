# Machine Learning

Some cool machine learning stuff. Provide code, explanations or whatever I might find interesting.

Author: Juan Maroñas Molano (jmaronasm@gmail.com) [PRHLT Research Center, Universidad Politécnica de Valencia]



## Install

* Python version 3.7
* Requirements: ``` pip install -r requirements.txt ```
* Might find useful to run ```./install.sh```



## Models Implemented

In folder models you can find different models, follow instructions there:

* models/Gaussian_Processes/ (regression and classification)

  * SVGP_TPG:
    *  stochastic sparse Variational GP    [ref](https://arxiv.org/abs/1309.6835) [ref](http://proceedings.mlr.press/v38/hensman15.pdf) 
    *  Transformed GP [ref](https://arxiv.org/abs/2011.01596)

* models/Bayesian_NN/ (only classification)

  * Mean Field Gaussian Variational BNN with pathwise gradient computations [ref](https://arxiv.org/abs/1505.05424)
* Mean Field Gaussian Variational BNN with local reparameterization [ref](https://arxiv.org/abs/1506.02557)   
  * Inference in Bayesian Neural Network with Hamiltonian Monte Carlo. Custom implementation in PyTorch [ref](https://arxiv.org/pdf/1206.1901.pdf)
  * Inference in a hierarchical Bayesian Neural Network using NUTS sampler. Implementation done in STAN
  * Point estimate Neural Network (Maximum Likelihood and Maximum Posterior)


## Other Stuff

In this folder I keep other things different to implementations.

* other/time_comparison_stan/
  * Keeps some time comparisons I have done with a model and the different possible implementations in stan

# Todo

* [ ] Add regression example to the Bayesian NN models. The ones comparing MFVI and HMC
* [ ] Stochastic gradient MCMC ( Hamiltonian Monte Carlo )  BNN [ref]()

##### Generative Models

I have a couple of generative models already implemented that perhaps I upload one day (will do when I need them for something, as I need to clean them up a bit):

* [ ] Probabilistic data augmentation using MCMC

* [ ] Probabilistic data augmentation with Mean Field VI (aka VAE)

* [ ] Probabilistic data augmentation with flows.

  

  

