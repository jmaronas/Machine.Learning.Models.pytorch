// ** Juan Maroñas **
// An example of Stan to perform inference in a Bayesian Neural Network
// This script does not use parallelism through reduce_sum
// It uses the categorical_glm function provided by STAN, which allow us to avoid performing the reduction of the likelihood using loops over N.

data {

      int<lower=1> N;     // number of draws from the model
      int<lower=2> C;     // number of output classes
      matrix[N,C]  x;     //  iid observations. X is a matrix of shape N,C where rows keeps the observations. Data is assumed to have same shape as classes
      int<lower=1> y[N];  //  iid observations. Y is an array of length N with type int containing the class label

      // definition of the prior p(w|0,I), passed as argument
      real          mu_w; 
      real<lower=0> std_w;

}

transformed data {
    vector[C] bias = rep_vector( 0.0, C ); // bias which in our model is going to be a vector of 0. Needed by categorical_logit_glm
}


parameters {
      matrix[C,C] W; // the parameter we want to infer
}

model {

      // prior over the weights p(w)
      profile("Prior"){ 
        to_vector(W) ~ normal(mu_w,std_w); 
      }
     
      // likelihood p(t|NNet_w(x)) 
      profile("Matrix product and likelihood"){
        target += categorical_logit_glm_lupmf( y | x, bias, W ); 
      }

    }


