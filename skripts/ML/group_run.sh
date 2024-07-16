#!/bin/bash
partition="cpu3-long"
time="14-00:00:00"
# declare -a algorithms=("Linear Discriminant Analysis" "Logistic Regression")
algorithms="K-neighbours classifier"
base_out="../../runs/ML/ml_"
tries=200
inner_fold=5
outer_fold=6
backend="tensorflow"
source_dir="../.."

# Com8 equal comb
source="Com8_equal_conc_comb"
file="20240702_FIA-Data_com8_equal_comb_mimedb_raw.xlsx"

sbatch --partition=$partition --time=$time --output="${base_out}annot_2.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "annotated" -sam $source -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}cos.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}maecos.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "gpu" -n "mae_cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}se.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "se_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}mae.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "mae_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}aecos.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "ae+cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}mse.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "mse_2" -fn "${file}" -a "${algorithms[@]}"


# COM 8 grown together
source="Com8_grown_together"
file="FIA-Data Com8_20230717_P0024_msAV206-312.xlsx"

sbatch --partition=$partition --time=$time --output="${base_out}gt_annot.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "annotated" -sam $source -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_cos.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "gpu" -n "cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_maecos.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "gpu" -n "mae_cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_se.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "se_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_mae.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "mae_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_aecos.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "ae+cosine_2" -fn "${file}" -a "${algorithms[@]}"
sbatch --partition=$partition --time=$time --output="${base_out}gt_mse.out" ML4com_run.py -t $tries -if $inner_fold -of $outer_fold -v 0 -st $source_dir -s "latent" -sam $source -b $backend -c "cpu" -n "mse_2" -fn "${file}" -a "${algorithms[@]}"