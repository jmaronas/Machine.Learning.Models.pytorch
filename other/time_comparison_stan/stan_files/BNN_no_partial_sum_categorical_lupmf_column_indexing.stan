// ** Juan Maroñas **
// An example of Stan to perform inference in a Bayesian Neural Network
// This script does not use parallelism through reduce_sum
// This script perform products so that column major order indexing can be performed

data {

      int<lower=1> N;     // number of draws from the model
      int<lower=2> C;     // number of output classes
      matrix[C,N]  x;     //  iid observations. X is a matrix of shape C,N where columns keeps the observations. Data is assumed to have same shape as classes
      int<lower=1> y[N];  //  iid observations. Y is an array of length N with type int containing the class label

      // definition of the prior p(w|0,I), passed as argument
      real          mu_w; 
      real<lower=0> std_w;

}


parameters {
      matrix[C,C] W; // the parameter we want to infer
}

model {

      // data declaration
      matrix[C,N] p; // so that it can be indexed column-wise

      // prior over the weights p(w)
      profile("Prior"){ 
      to_vector(W) ~ normal(mu_w,std_w); 
      }
     
      // likelihood p(t|NNet_w(x)) 
      profile("Matrix product"){
      p = W*x; // get parameters from Categorical Likelihood. Operation done in this way for column major indexing
      }

      profile("Likelihood evaluation"){
      for (n in 1:N)
         target += categorical_logit_lupmf( y[n] | p[:,n]); // indexing by columns, which is faster
      }
    }


