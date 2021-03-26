// ** Juan Maroñas **
// An example of Stan to perform inference in a Bayesian Neural Network

// Within chain parallelism is achieved through reduce_sum. In this case
// rather than copying the result of the product p=x*W into the function
// the parameter W is copied. Then we use the GLM function which allow us
// avoid the loop over the data. Moreover, data x is passed following the standard 
// format. This means that indexing is performed row wise which will be less efficient.


functions {
  
  real partial_sum_lpmf(int [] y_slice, int start, int end, matrix W, matrix x, vector bias ) 
  {

    return categorical_logit_glm_lupmf(y_slice | x[start:end], bias, W);

  }

  
  }

data {

      // It is usefull to set the lower or upper values for the variables. This improves readiability
      int<lower=0> N;     // number of draws from the model
      int<lower=2> C;     // number of logits, i.e number of classes
      matrix[N,C]  x;     //  iid observations. X is a matrix with data given by rows
      int<lower=1> y[N];  //  iid observations. Y is an array of length N containing the class label

      // definition of the prior p(w|0,I), passed as argument
      real          mu_w; 
      real<lower=0> std_w;

}

transformed data{

      // data declaration
      int grainsize = 1;

      // bias vector
      vector [C] bias = rep_vector(0.0, C);
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
         target += reduce_sum(partial_sum_lupmf, y, grainsize, W , x, bias);
      }
}


