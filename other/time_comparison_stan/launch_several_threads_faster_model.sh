## Test the fastest model with several threads

compile_openmpi='' # leave blank; no difference in compiling with openmpi as reduce_sum does not use it

compile_gpu='--compile_gpu' # leave blank for cpu, set '--compile_gpu' to compile in GPU

## Models that use reduce_sum
threads=(1 2 4 8 10 16 20 24 32)

for th in "${threads[@]}"
do

    stan_threads="--stan_threads "$th

    for i in {1..5}
    do
        model_type='partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_Categorical_GLM_columns_indexing'
        python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

        model_type='partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_Categorical_GLM_row_indexing'
        python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

        sleep 1m

    done

    cd stan_files
    ls  | grep -v .stan | xargs rm
    cd ..


done




    






