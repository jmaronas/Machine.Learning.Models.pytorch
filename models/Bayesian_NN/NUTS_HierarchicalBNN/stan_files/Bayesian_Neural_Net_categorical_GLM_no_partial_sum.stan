// ** Juan Maroñas **
// An example of Stan to perform inference in a hierarchical Bayesian Neural Network. The hierarchical distribution is similar to that used in Radford Neal's thesis, i.e a standard prior over the 
// parameters where the variance is controlled through a inverse gamma distribution. Also, rather than using Gibbs sampling for the hyperparameter (as Radford), Stan performs HMC
// inference over everything.

// This script does not use parallelism through reduce_sum. Check file BNN_partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_Categorical_GLM_columns_indexing.stan 
// The Bayesian Neural Network uses relu activation function and the size of the hidden layers is fixed, i.e all hidden layers have same number of neurons


functions
{

   matrix relu(matrix x, matrix zeros)
   {
       return fmax(x, zeros);
   }

}

data {

      int<lower=1> N;     // number of draws from the model
      int<lower=2> C;     // number of output classes
      matrix[N,C]  x;     //  iid observations. X is a matrix of shape N,C where rows keeps the observations. Data is assumed to have same shape as classes
      int<lower=1> y[N];  //  iid observations. Y is an array of length N with type int containing the class label

      // definition of the neural network topology
      int<lower=0> num_hidden;  // number of hidden layers. 0 will perform a logistic regression
      int<lower=1> num_neurons; // number of neurons per layer

      // definition of the prior p(w|0,\sigma^2 I), passed as argument. The variance has an hyperprior given by inverse_gamma
      real mu_w; 

}

transformed data {
    
    matrix[N,num_neurons] auxiliary_zeros_hidden; // Auxiliary variable used to compute relu function. Declared here to avoid creating the vector on the fly each time

    int cond_input  = num_hidden == 0;
    int cond_hidden = num_hidden >  1;

}


// Parameters sampled by stan

parameters {
      // hyperparameters
      real<lower=0> sigma; 

      // parameters

      // input layer
      matrix[C,   cond_input ? C : num_neurons] Winp;
      row_vector[ cond_input ? C : num_neurons] binp;

      // output layer
      matrix[    cond_input ? 0 : num_neurons, cond_input ? 0 : C] Wout;  // projection from output layer
      vector[cond_input ? 0 : C]                                   bout;  // bias from output layer. The only one that is declared as vector

      // hidden layer
      matrix[cond_hidden ? num_neurons : 0, cond_hidden ? num_neurons : 0] Wh[cond_hidden ? num_hidden-1 : 0]; // projections from hidden layers
      row_vector[cond_hidden ? num_neurons :0 ]                            bh[cond_hidden ? num_hidden-1 : 0]; // bias from hidden layer
        
}



model {
    
      // auxiliary variables
      matrix[N,num_hidden] ph;

      // hyperprior over the precision from the prior over the parameters
      sigma ~ inv_gamma(1.0,1.0); // make it weakly non-informative

      real std_w = sqrt(sigma); // associated standard deviation

      // prior over the weights p(w)
      profile("Prior"){ 

          // input layer
          to_vector(Winp) ~ normal(mu_w,std_w); 
          binp            ~ normal(mu_w,std_w);
      
          if (num_hidden > 0)
          {
              // output layer
              to_vector(Wout) ~ normal(mu_w,std_w);
              bout            ~ normal(mu_w,std_w);

              if (num_hidden > 1 ) 
              {
                for (n in 1:num_hidden-1)
                {
                  // hidden layer
                  to_vector(Wh[n]) ~ normal(mu_w,std_w);
                  bh[n]            ~ normal(mu_w,std_w);
                }
              }
          }

      }
     
      // likelihood p(t|NNet_w(x)) 
      profile("Matrix product and likelihood"){
     
          if (num_hidden == 0)
            target += categorical_logit_glm_lupmf(y | x, binp', Winp);

          else
          {
              ph = relu( x*Winp + rep_matrix(binp,N), auxiliary_zeros_hidden );

             if (num_hidden > 1 )
             {
                  for (n in 1:num_hidden-1)
                    ph = relu( ph*Wh[n] + rep_matrix(bh[n],N) , auxiliary_zeros_hidden );
             }

              target += categorical_logit_glm_lupmf( y | ph, bout, Wout ); 

          }
      }

    }


