// ** Juan Maroñas **
// An example of Stan to perform inference in a Bayesian Neural Network
// This script uses within chain parallelism through reduce_sum
// In this case we slice over data, rather than parameters (this means
// that parameters p will be copied each time). On the other hand, slicing over p
// is done using column major order to allow efficient indexing of the matrix p.


functions {
  
  real partial_sum_lpmf(int [] y_slice, int start, int end, matrix p ) 
  {

    real __target = 0; 
    int counter   = 1;
    for (n in start:end)
    {
      __target += categorical_logit_lupmf( y_slice[counter] | p[:,n]); 
      counter  += 1;
    }

    return __target;
  }

  
  }

data {

      // It is usefull to set the lower or upper values for the variables. This improves readiability
      int<lower=0> N;     // number of draws from the model
      int<lower=2> C;     // number of logits, i.e number of classes
      matrix[C,N]  x;     //  iid observations. X is a C,N matrix with data given by columns
      int<lower=1> y[N];  //  iid observations. Y is an array of length N containing the class label

      // definition of the prior p(w|0,I), passed as argument
      real          mu_w; 
      real<lower=0> std_w;

}

transformed data{

      // data declaration
      int grainsize = 1;
  }

parameters {
      matrix[C,C] W; // the parameter we want to infer
}

model {

      matrix[C,N] p; // so that it can be indexed column-wise

      // prior over the weights p(w)
      profile("Prior"){ 
         to_vector(W) ~ normal(mu_w,std_w); 
      }
      
      // likelihood p(t|NNet_w(x)) 
      profile("Matrix product"){
         p = W*x; // get parameters from Categorical Likelihood. No need for transposition
      }

      // parallelize the computation
      profile("Likelihood evaluation"){
         target += reduce_sum(partial_sum_lupmf, y, grainsize, p);
      }
}


