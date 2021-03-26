// ** Juan Maroñas **
// An example of Stan to perform inference in a Bayesian Neural Network
// This script uses within chain parallelism through reduce_sum
// To reduce computational cost, parameters p = x*W are sliced over.
// However, rather than using column indexing, it uses row_indexing at the matrix level

functions {
  
  real partial_sum_lpdf( row_vector [] p_slice, int start, int end, int [] y ) 
  {

    real __target = 0; 
    int counter   = 1;
    for (n in start:end)
    {
      __target += categorical_logit_lupmf( y[n] | p_slice[counter]');  // transpose here as expecting column vector
      counter  += 1;
    }

    return __target;
  }

  
  }

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


transformed data{

      // data declaration
      int grainsize = 1;
  }

parameters {
      matrix[C,C] W; // the parameter we want to infer
}

model {

      matrix[N,C] p;          // this is indexed row-wise
      row_vector[C] p_vec[N]; // use array of row vectors so that matrix is reshaped into this for slicing.

      // prior over the weights p(w)
      profile("Prior"){ 
         to_vector(W) ~ normal(mu_w,std_w); 
      }
      
      // likelihood p(t|NNet_w(x)) 
      profile("Matrix product"){
         p = x*W; // get parameters from Categorical Likelihood. No need for transposition
      }

      // reshape into array of column_vectors to be sliced
      profile("Matrix reshape into vector"){
      for (n in 1:N)
        p_vec[n] = p[n]; // matrix is sliced over rows, which is more ineficient
      }


      // parallelize the computation
      profile("Likelihood evaluation"){
         target += reduce_sum(partial_sum_lupdf, p_vec, grainsize, y);
      }
}


