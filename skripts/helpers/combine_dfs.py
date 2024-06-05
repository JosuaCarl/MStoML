#!/usr/bin/env python3
#SBATCH --mem=400G

# imports
import os
import shutil
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

import argparse

def read_df(path, framework=pd):
    print(f"Loading: {path}")
    if path.endswith(".parquet"):
        df = framework.read_parquet( path )
    elif path.endswith(".tsv"):
        if framework == pl:
            df = framework.read_csv( path, separator="\t" )
        elif framework == pd:
            df = framework.read_csv( path, sep="\t", index_col="mz")
    elif path.endswith(".feather"):
        df = framework.read_feather( path )
    return df

def write_df(df, path, framework=pd):
    if framework == pl:
        if path.endswith(".parquet"):
            df.write_parquet( path )
        elif path.endswith(".tsv"):
            df.write_csv( path, separator="\t" )
    elif framework == pd:
        if path.endswith(".parquet"):
            df.to_parquet( path )
        elif path.endswith(".tsv"):
            df.to_csv( path, sep="\t" )
        elif path.endswith(".feather"):
            df.to_feather( path )

def concat_dfs(dfs, framework=pd):
    if framework == pl:
        dfs = framework.concat( dfs, how="align" )
    elif framework == pd:
        dfs = framework.concat( dfs, axis="columns" )
    return dfs


def combine_dc(path_combs, outpath, target_format="parquet", framework=pl, bins:int=2):
    if len(path_combs) == 1:
        if path_combs[0].endswith(target_format):
            shutil.copy(path_combs[0], os.path.join(outpath, f"data_matrix.{target_format}"))
        else:
            binned_df = read_df( path_combs[0], framework=framework)
            write_df(binned_df, os.path.join(outpath, f"data_matrix.{target_format}"), framework=framework)
    
    else:
        tmp_dir = os.path.join(outpath, "tmp")
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        tmp_paths = []
        for i, path in enumerate(tqdm(path_combs)):
            file = "" if os.path.isfile(path) else "data_matrix.tsv"
            path =  os.path.normpath(os.path.join(path, file))
            split = str(os.path.basename(path)).split(".")
            target_file = f'{".".join( split[:-1] )}_{i}.{target_format}'
            tmp_paths.append(os.path.join(tmp_dir, target_file))

        binned_dfs = []
        new_path_combs = []
        for i, path in enumerate(tqdm(path_combs)):
            check_previous_runs = [os.path.isfile(tmp_path) for tmp_path in tmp_paths[i:np.min([i + bins, len(tmp_paths) - 1])]]
            if True not in check_previous_runs:
                binned_df = read_df(path, framework=framework)
                binned_dfs.append( binned_df )
        
                if len(binned_dfs) >= bins:
                    binned_dfs = concat_dfs(binned_dfs, framework=framework)
                    write_df(binned_dfs, tmp_paths[i], framework=framework)
                    new_path_combs.append( tmp_paths[i] )
                    binned_dfs = []
                
        if binned_dfs and not os.path.isfile(tmp_paths[i]):
            binned_dfs = concat_dfs(binned_dfs, framework=framework)
            write_df(binned_dfs, tmp_paths[i], framework=framework)
            new_path_combs.append( tmp_paths[i] )
    
        print(new_path_combs)
        combine_dc(new_path_combs, outpath, target_format=target_format, framework=framework)



def main(args):
    in_dir = args.in_dirs
    out_dir = args.out_dir
    target_format = args.target_format
    bins = args.bins
    combine_dc([os.path.join(in_dir, file) for file in os.listdir(in_dir) if file.endswith(".tsv")], out_dir, target_format=target_format, framework=pd, bins=bins)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                description='Hyperparameter tuning for Variational Autoencoder with SMAC')
    parser.add_argument('-i', '--in_dirs', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument('-t', '--target_format', required=True)
    parser.add_argument('-b', '--bins', type=int, required=True)
    args = parser.parse_args()

    main(args)
