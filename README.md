# Machine Learning

Some cool machine learning stuff. Provide code, explanations or whatever I might find interesting.

Author: Juan Maroñas Molano (jmaronasm@gmail.com) [PRHLT Research Center, Universidad Politécnica de Valencia]



## Install

* Python version 3.7
* Requirements: ``` pip install -r requirements.txt ```
* Might find useful to run ```./install.sh```



## Models Implemented

In folder models you can find different models, follow instructions there:

* models/Bayesian_NN/

  * Mean Field Gaussian Variational BNN with pathwise gradient computations [ref](https://arxiv.org/abs/1505.05424)

  * Mean Field Gaussian Variational BNN with local reparameterization [ref](https://arxiv.org/abs/1506.02557)   

# Todo

Clean up and refactorize code for:

* [ ] Add GP and TGP
* [ ] Stochastic gradient MCMC ( Hamiltonian Monte Carlo )  BNN [ref](https://arxiv.org/abs/1206.1901), for classification
* [ ] Point Estimate NN (should not go here but useful for comparison), for classification
* [ ] Probabilistic data augmentation using MCMC
* [ ] Probabilistic data augmentation with Mean Field VI (aka VAE)
* [ ] Probabilistic data augmentation with flows.
* [ ] Add regression example to the Bayesian NN models. The ones comparing MFVI and HMC
