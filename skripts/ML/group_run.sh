#!/bin/bash
: '
Run multiple groups of Machine Learning Tuning, Training or Evaluation experiments on a slurm cluster.
'

partition="cpu1"
time="1-00:00:00"
declare -a algorithms=( "Decision tree" "Gaussian Naive-Bayes" "K-neighbours classifier" "Logistic Regression" "Neural Network (MLP) SK-learn" "Random forest" "Support-vector classifier" "Extreme gradient boosting RF" "Linear Discriminant Analysis" ) # 
base_out="../../runs/ML/ml_"
tries=200
inner_fold=5
outer_fold=6
workers=1
backend="tensorflow"
source_dir="../.."
suffix=""
task="evaluate"

: '
# Com8 equal comb
source="Com8_equal_conc_comb"
model_source="Com8_grown_together"
file="20240702_FIA-Data_com8_equal_comb_mimedb_raw.xlsx"
model_file="FIA-Data Com8_20230717_P0024_msAV206-312.xlsx"

sbatch --partition=$partition --cpus-per-task=1 --ntasks-per-node=$workers --time=$time --output="${base_out}ec_annot${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "annotated" -sam $source -ms $model_source \
    -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers


sbatch --partition=$partition --cpus-per-task=1 --ntasks-per-node=$workers --time=$time --output="${base_out}ec_cos${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

sbatch --partition=$partition --cpus-per-task=1 --ntasks-per-node=$workers --time=$time --output="${base_out}ec_maecos${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "gpu" -n "mae_cosine_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

sbatch --partition=$partition --cpus-per-task=1 --ntasks-per-node=$workers --time=$time --output="${base_out}ec_se${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "cpu" -n "se_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

sbatch --partition=$partition --cpus-per-task=1 --ntasks-per-node=$workers --time=$time --output="${base_out}ec_mae${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "cpu" -n "mae_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

sbatch --partition=$partition --cpus-per-task=1 --ntasks-per-node=$workers --time=$time --output="${base_out}ec_aecos${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "cpu" -n "ae+cosine_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

sbatch --partition=$partition --cpus-per-task=1 --ntasks-per-node=$workers --time=$time --output="${base_out}ec_mse${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "cpu" -n "mse_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

'

# COM 8 grown together
source="Com8_grown_together"
model_source="Com8_equal_conc_comb"
file="FIA-Data Com8_20230717_P0024_msAV206-312.xlsx"
model_file="20240702_FIA-Data_com8_equal_comb_mimedb_raw.xlsx"

sbatch --partition=$partition --time=$time --output="${base_out}gt_annot${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "annotated" -sam $source -ms $model_source \
    -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

sbatch --partition=$partition --time=$time --output="${base_out}gt_cos${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

sbatch --partition=$partition --time=$time --output="${base_out}gt_maecos${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "gpu" -n "mae_cosine_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

sbatch --partition=$partition --time=$time --output="${base_out}gt_se${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "cpu" -n "se_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

sbatch --partition=$partition --time=$time --output="${base_out}gt_mae${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "cpu" -n "mae_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

sbatch --partition=$partition --time=$time --output="${base_out}gt_aecos${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "cpu" -n "ae+cosine_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers

sbatch --partition=$partition --time=$time --output="${base_out}gt_mse${task}${suffix}.out" \
    ML4com_run.py \
    -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -ms $model_source \
    -b $backend -c "cpu" -n "mse_2" -fn "${file}" -mfn "${model_file}" -a "${algorithms[@]}" -ta $task -w $workers
