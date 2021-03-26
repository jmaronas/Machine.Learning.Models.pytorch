compile_openmpi='' # leave blank; no difference in compiling with openmpi as reduce_sum does not use it
compile_gpu='' # leave blank for cpu, set '--compile_gpu' to compile in GPU

## Models that do not use reduce_sum
for i in {1..5}
do

    model_type='no_partial_sum_categorical_GLM'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    model_type='no_partial_sum_categorical_lupmf_column_indexing'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    model_type='no_partial_sum_categorical_lupmf_column_indexing_with_transposition'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    model_type='no_partial_sum_categorical_lupmf_row_indexing'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    sleep 1m # needed so that csv files output by cmdstan are not overwritten

done

## Models that use reduce_sum

stan_threads="--stan_threads 10" # leave blank ''  so that no threads are used, i.e 1 thread is used

for i in {1..5}
do

    model_type='partial_sum_SLICED_ARGS_logit_SHARED_ARGS_y_row_indexing_and_transposition'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    model_type='partial_sum_SLICED_ARGS_logit_SHARED_ARGS_y'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    model_type='partial_sum_SLICED_ARGS_y_SHARED_ARGS_logit_column_indexing'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    model_type='partial_sum_SLICED_ARGS_y_SHARED_ARGS_logit_column_indexing_with_transposition'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    model_type='partial_sum_SLICED_ARGS_y_SHARED_ARGS_logit_row_indexing'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    model_type='partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_Categorical_GLM_columns_indexing'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    model_type='partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_Categorical_GLM_row_indexing'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    model_type='partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_column_indexing'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    model_type='partial_sum_SLICED_ARGS_y_SHARED_ARGS_W_row_indexing'
    python BNN_time_comparison.py --model_type $model_type $compile_gpu $compile_openmpi $stan_threads

    sleep 1m

done





    






