### Time comparison in Stan

This folder keeps an inference time comparison  of different implementation of the same model in STAN. The model is given by a linear logistic regression with a constant bias. The parameter posterior infered is given by:
$$
p(w|Y,X) \propto \prod^N_n \text{Cat}(y_n|Wx_n +0)\mathcal{N}(W|0,I)
$$
The Stan forum thread that follows this comparison is given [here](https://discourse.mc-stan.org/t/measuring-and-comparing-computational-performance-in-stan-with-different-compilation-alternatives-using-reduce-sum-does-not-bring-any-advantage/21340/26). What motivated me to perform the following comparison was the fact that I wanted to test how parallelism work with the `reduce_sum` function from Stan, and got very bad performance results. Thanks to the thread, to @wds15 and to this [video](https://www.youtube.com/watch?v=d5gPjajxN9A&list=PLCrWEzJgSUqwychgV5Q72Nsaq08FU48aQ) I was able to extract the juice of how make an efficient implementation in Stan. I have already open an issue in Stan documentation so that they provide more details on how the `reduce_sum` function works and how it handles the input arguments (for example shared parameters are always copied while sliced parameters are only copied once), so that practitioners can use this function efficiently.

This implementation checks different ways of performing the log likelihood evaluation, GPU vs CPU, row vs column indexing,  how different arguments to the `reduce_function` affects computation, and how different threads improve the performance of the within chain paralleization. It does it for a relatively large amount of data $N=45000$ , although this can be changed in the python script (see below).

The time comparisons are provided in the `time_comparison.ods` libre office book. This book contains two sheets. The first one compares the performance of different implementations using CPU vs GPU, 1 vs 10 Stan threads and more. Note that I first performed a comparison with openMPI compilation. Then, @wds15 point me to the fact that `reduce_sum` does not use this library (hopefully this information will be included in Stan docs). The time results provided are average of 5 runs.

This repo contains the following things:

* `stan_files`: folder containing the different implementations of Stan models. Up to 13 models.
* `results_time_comparison`: holds the files obtained by launching Stan that compare the different implementations.
* `results_different_threads`: holds the files obtained by launching Stan that compare different number of threads with the fastest model obtained in the comparison
* `BNN_time_comparison.py`: python file that uses `cmdstanpy` to launch Stan programs.  Execute `python BNN_time_comparison.py --help` to check for arguments. The dataset size $N$ can be modified in this file.
* Bash files: keep the bash files that can be used to launch different test. Note that these file launches the same experiments 5 times.

**NOTE:** The name of the experiment in the excel book is the same as the name of the Stan file that launches a particular implementation. Each stan file has a header providing an overview of what specific things from Stan that file tests. Also, to launch that particular experiment, the python code takes as argument the same name as the stan file, to make it easier to follow. Check the bash files for examples.

To take the average of the K runs, you can just use the following bash line:

`ls | grep stdout | xargs cat | grep "1000 transitions using 10 leapfrog steps per transition would take" | awk 'BEGIN{acc=0.0;norm=0.0}{acc+=$11;norm+=1}END{print acc/norm}'`

