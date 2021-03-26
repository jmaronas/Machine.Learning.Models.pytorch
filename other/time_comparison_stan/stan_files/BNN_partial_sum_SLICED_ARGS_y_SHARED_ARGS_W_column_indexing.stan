// ** Juan Maroñas **
// An example of Stan to perform inference in a Bayesian Neural Network

// Within chain parallelism is achieved through reduce_sum. In this case
// rather than copying the result of the product p=x*W into the function
// the parameter W is copied. Then the matrix product of p=W*x is performed
// by indexing x column wise over the batch that is performed by each partial
// sum. Finally, the result of the operation is passed into a categorical
// likelihood.



functions {
  
  real partial_sum_lpmf(int [] y_slice, int start, int end, matrix W, matrix x, int C ) 
  {

    real __target = 0; 
  
    matrix[C, end + start + 1] p; // so that it can be indexed column-wise
    p = W*x[:,start:end];

    for (n in 1:(end-start+1))
    {
      __target += categorical_logit_lupmf( y_slice[n] | p[:,n]); 
    }

    return __target;
  }

  
  }

data {

      // It is usefull to set the lower or upper values for the variables. This improves readiability
      int<lower=0> N;     // number of draws from the model
      int<lower=2> C;     // number of logits, i.e number of classes
      matrix[C,N]  x;     //  iid observations. X is a matrix with data given by columns
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


      // prior over the weights p(w)
      profile("Prior"){ 
         to_vector(W) ~ normal(mu_w,std_w); 
      }
      
      // likelihood p(t|NNet_w(x)) 
      // performed inside the reduce sum

      // parallelize the computation
      profile("Likelihood evaluation"){
         target += reduce_sum(partial_sum_lupmf, y, grainsize, W , x, C);
      }
}


