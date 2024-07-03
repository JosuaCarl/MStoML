# Com8 equal comb
sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_annot.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "annotated" -sam "Com8_equal_conc_comb" -b "tensorflow" -c "gpu" -n "cosine_2" -fn "20240702_FIA-Data_com8_equal_comb_mimedb_raw.xlsx"

sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_cos.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_equal_conc_comb" -b "tensorflow" -c "gpu" -n "cosine_2" -fn "20240702_FIA-Data_com8_equal_comb_mimedb_raw.xlsx"
sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_maecos.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_equal_conc_comb" -b "tensorflow" -c "gpu" -n "mae_cosine_2" -fn "20240702_FIA-Data_com8_equal_comb_mimedb_raw.xlsx"
sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_se.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_equal_conc_comb" -b "tensorflow" -c "cpu" -n "se_2" -fn "20240702_FIA-Data_com8_equal_comb_mimedb_raw.xlsx"
sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_mae.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_equal_conc_comb" -b "tensorflow" -c "cpu" -n "mae_2" -fn "20240702_FIA-Data_com8_equal_comb_mimedb_raw.xlsx"

#sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_aecos.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_equal_conc_comb" -b "tensorflow" -c "cpu" -n "ae+cosine_2" -fn "20240702_FIA-Data_com8_equal_comb_mimedb_raw.xlsx"
#sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_mse.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_equal_conc_comb" -b "tensorflow" -c "cpu" -n "mse_2" -fn "20240702_FIA-Data_com8_equal_comb_mimedb_raw.xlsx"



# COM 8 grown together
sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_gt_annot.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "annotated" -sam "Com8_grown_together" -b "tensorflow" -c "gpu" -n "cosine_2" -fn "20230717_FIA-Data_P0024_msAV206-413.xlsx"

sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_gt_cos.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_grown_together" -b "tensorflow" -c "gpu" -n "cosine_2" -fn "20230717_FIA-Data_P0024_msAV206-413.xlsx"
sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_gt_maecos.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_grown_together" -b "tensorflow" -c "gpu" -n "mae_cosine_2" -fn "20230717_FIA-Data_P0024_msAV206-413.xlsx"
sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_gt_se.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_grown_together" -b "tensorflow" -c "cpu" -n "se_2" -fn "20230717_FIA-Data_P0024_msAV206-413.xlsx"
sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_gt_mae.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_grown_together" -b "tensorflow" -c "cpu" -n "mae_2" -fn "20230717_FIA-Data_P0024_msAV206-413.xlsx"

#sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_gt_aecos.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_grown_together" -b "tensorflow" -c "cpu" -n "ae+cosine_2" -fn "20230717_FIA-Data_P0024_msAV206-413.xlsx"
#sbatch --partition=cpu3-long --time=14-00:00:00 --output ../../runs/ML/ml_gt_mse.out ML4com_run.py -t 200 -if 5 -of 6 -v 0 -st "../.." -s "latent" -sam "Com8_grown_together" -b "tensorflow" -c "cpu" -n "mse_2" -fn "20230717_FIA-Data_P0024_msAV206-413.xlsx"
