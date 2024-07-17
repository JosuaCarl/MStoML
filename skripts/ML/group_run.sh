#!/bin/bash
partition="cpu3-long"
time="14-00:00:00"
# declare -a algorithms=("Linear Discriminant Analysis" "Logistic Regression")
algorithms="Neural Network (MLP) SK-learn"
base_out="../../runs/ML/ml_"
tries=200
inner_fold=5
outer_fold=6
backend="tensorflow"
source_dir="../.."
suffix="_svc"

# Com8 equal comb
source="Com8_equal_conc_comb"
file="20240702_FIA-Data_com8_equal_comb_mimedb_raw.xlsx"

sbatch --partition=$partition --time=$time --output="${base_out}ec_annot${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "annotated" -sam $source -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}ec_cos${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}ec_maecos${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "gpu" -n "mae_cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}ec_se${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "se_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}ec_mae${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "mae_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}ec_aecos${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "ae+cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}ec_mse${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "mse_2" -fn "${file}" -a "${algorithms[@]}"


# COM 8 grown together
source="Com8_grown_together"
file="FIA-Data Com8_20230717_P0024_msAV206-312.xlsx"

sbatch --partition=$partition --time=$time --output="${base_out}gt_annot${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "annotated" -sam $source -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_cos${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_maecos${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "gpu" -n "mae_cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_se${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "se_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_mae${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "mae_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_aecos${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "ae+cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_mse${suffix}.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "mse_2" -fn "${file}" -a "${algorithms[@]}"