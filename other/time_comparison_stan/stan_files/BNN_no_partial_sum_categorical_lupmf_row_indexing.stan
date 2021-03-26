// ** Juan Maroñas **
// An example of Stan to perform inference in a Bayesian Neural Network
// This script does not use parallelism through reduce_sum
// This script perform products so that row major indexing is performed. This is the standard matrix product performed in Neural Nets with standard data representation. However this
// type of indexing is ineficient in Stan.

data {

      // It is usefull to set the lower or upper values for the variables. This improves readiability
      int<lower=0> N;     // number of draws from the model
      int<lower=2> C;     // number of logits, i.e number of classes
      matrix[N,C]  x;     //  iid observations. X is a matrix of shape N,C where rows keep data.
      int<lower=1> y[N];  //  iid observations. Y is an array of length N containing the class label

      // definition of the prior p(w|0,I), passed as argument
      real          mu_w; 
      real<lower=0> std_w;

}

parameters {
      matrix[C,C] W; // the parameter we want to infer
}

model {
      // data declaration
      matrix[N,C] p; // indexed row_wise

      // prior over the weights p(w)
      profile("Prior"){ 
      to_vector(W) ~ normal(mu_w,std_w); 
      }
      
     
      // likelihood p(t|NNet_w(x)) 
      profile("Matrix product"){
      p = x*W; // get parameters from Categorical Likelihood. Standard operation in NNet
      }

      profile("Likelihood evaluation"){
      for (n in 1:N)
         target += categorical_logit_lupmf( y[n] | p[n]'); // need a final transposition as likelihood function expects column vector type
      }
    }


