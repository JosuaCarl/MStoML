#!/usr/bin/env python3
#SBATCH --job-name VAE_interpret
#SBATCH --mem 100G
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1

import sys
import os
import seaborn as sns
sys.path.append( '..' )

from VAE.smac_runhistories import *
from VAE.vae import *

# reports_dir  "/mnt/d/reports/VAE"
# results_dir  "/mnt/d/runs/VAE/results"
# data_dir  "/mnt/d/runs/FIA/Com8_grown_together/merged"
# training_dir "/mnt/d/runs/VAE/training"
# backend_name = "tensorflow"
# computation = "cpu"
# name = "mae_2"
# project = f"vae_{backend_name}_{computation}_{name}"


def main(args):
    # Define dictionaries
    data_dir, results_dir, reports_dir, training_dir = [os.path.normpath(os.path.join(os.getcwd(), d)) for d in  [args.data_dir, args.results_dir, args.reports_dir, args.training_dir]]

    # Define computation parameters of model
    backend_name = args.backend
    computation = args.computation
    name = args.name if args.name else None
    project = f"vae_{backend_name}_{computation}_{name}" if name else f"vae_{backend_name}_{computation}"

    verbosity =  args.verbosity if args.verbosity else 0

    model_dir = Path(os.path.normpath( os.path.join(training_dir, project)))
    outdir = os.path.normpath(os.path.join(reports_dir, project))

    # Load model
    model = keras.saving.load_model(os.path.join(model_dir, f"{project}_best.keras"), custom_objects=None, compile=True, safe_mode=True)

    if verbosity >= 1:
        model.summary()

    # Data load
    if "*" in data_dir:
        for source in ["Com8_grown_together", "Com8_equal_conc_comb"]:
            encode_reconstruct(model=model, data_dir=data_dir.replace("*", source), results_dir=results_dir, name=name, source=source, verbosity=verbosity)
    else:
        encode_reconstruct(model=model, data_dir=data_dir, results_dir=results_dir, name=name, source="", verbosity=verbosity)


def encode_reconstruct(model, data_dir, results_dir, name, source, verbosity):
    outdir = os.path.join(results_dir, source)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    X = read_data(data_dir, verbosity=verbosity)

    # Latent space construction
    vae_enc = pd.DataFrame( model.encode_mu(X) )
    print(f"Shape of latent dimension: {vae_enc.shape}")
    vae_enc.to_csv( os.path.join(outdir, f"encoded_mu_{name}.tsv"), sep="\t" )

    ax = sns.lineplot(data=vae_enc[:5].T)
    plt.xlabel("Latent dimension")
    plt.ylabel("Value")
    plt.savefig( os.path.join(outdir, f"encoded_mu_5_{name}.png") )
    plt.close()

    # Reconstruction of data for visualization
    reconstructed_data = total_ion_count_normalization( pd.DataFrame( model(X).numpy() ), axis=1)
    reconstructed_data.to_csv( os.path.join(outdir, f"reconstructed_data_{name}.tsv"), sep="\t" )

    scale = 1.0
    plot_df = pd.DataFrame(reconstructed_data.loc[0].values * scale , index=X.columns, columns=["inty"]).reset_index()    # Adjustment by scale
    plot_df_2 = pd.DataFrame(X.iloc[0].values, index=X.columns, columns=["inty"]).reset_index()

    mae = np.mean( np.abs((plot_df_2["inty"] - plot_df["inty"])) )
    ae = np.sum( np.abs((plot_df_2["inty"] - plot_df["inty"])) )

    sns.set_theme(rc={"figure.figsize":(12, 6)})
    ax = sns.lineplot(data=plot_df_2, x="mz", y="inty", alpha=0.7, label="Original spectrum")
    predicted_label = "Predicted spectrum" if scale == 1.0 else f"Predicted spectrum (Scale: {scale})"
    ax = sns.lineplot(data=plot_df, x="mz", y="inty", alpha=0.7, label=predicted_label)
    plt.ylabel("Intensity")
    plt.xlabel("m/z")
    plt.annotate(f"MAE: {format(mae, '.12f')}", xy=(0.795, 0.79), xycoords='axes fraction', fontsize=11, fontfamily="Monospace")
    plt.annotate(f" AE: {format(ae, '.12f')}", xy=(0.795, 0.74), xycoords='axes fraction', fontsize=11, fontfamily="Monospace")
    plt.legend()
    plt.savefig( os.path.join(outdir, f"reconstruction_original_{name}.png") )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='VAE_interpret',
                                     description='Interpretation of VAE training and Extraction of Encoding + Reconstruction')
    
    parser.add_argument('-d', '--data_dir', required=True)
    parser.add_argument('-res', '--results_dir', required=True)
    parser.add_argument('-rep', '--reports_dir', required=True)
    parser.add_argument('-t', '--training_dir', required=True)

    parser.add_argument('-b', '--backend', required=True)
    parser.add_argument('-c', '--computation', required=True)
    parser.add_argument('-n', '--name', required=False)

    parser.add_argument('-v', '--verbosity', type=int, required=False)
    args = parser.parse_args()

    main(args)
