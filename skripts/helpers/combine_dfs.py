#!/usr/bin/env python3

# imports
import os
import mat73
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
            df = framework.read_csv( path, sep="\t" )
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
        dfs = framework.concat( dfs )
    return dfs


def combine_dc(path_combs, outpath, target_format="parquet", framework=pl):
    if len(path_combs) == 1:
        binned_df = read_df( path_combs[0], framework=framework)
        write_df(binned_df, os.path.join(outpath, f"data_matrix.{target_format}"), framework=framework)
        return binned_df
    
    else:
        tmp_dir = os.path.join(outpath, "tmp")
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        binned_dfs = []
        new_path_combs = []
        for i, path in enumerate(tqdm(path_combs)):
            file = "" if os.path.isfile(path) else "data_matrix.tsv"
            path =  os.path.normpath(os.path.join(path, file))
            split = str(os.path.basename(path)).split(".")
            target_file = f'{".".join( split[:-1] )}_{i // 2}.{target_format}'
            tmp_path = os.path.join(tmp_dir, target_file)
            if tmp_path not in new_path_combs:
                new_path_combs.append( tmp_path )
            if os.path.isfile(tmp_path):
                continue

            binned_df = read_df(path, framework=framework)
            binned_dfs.append( binned_df )
    
            if len(binned_dfs) >= 2:
                binned_dfs = concat_dfs(binned_dfs, framework=framework)
                write_df(binned_dfs, tmp_path, framework=framework)
                binned_dfs = []
                
        if binned_dfs:
            binned_dfs = concat_dfs(binned_dfs, framework=framework)
            write_df(binned_df, tmp_path, framework=framework)
    
    combine_dc(new_path_combs, outpath, target_format=target_format, framework=framework)


def main(args):
    in_dir = args.in_dirs
    out_dir = args.out_dir
    combine_dc([os.path.join(in_dir, file) for file in os.listdir(in_dir) if file.endswith(".tsv")], out_dir, target_format="feather", framework=pd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                description='Hyperparameter tuning for Variational Autoencoder with SMAC')
    parser.add_argument('-i', '--in_dirs', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    args = parser.parse_args()

    main(args)
