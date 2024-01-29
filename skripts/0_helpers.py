# imports
import os
import mat73
import pandas as pd

def parse_folder(path):
    return os.listdir(path)


def mat_to_tsv(folder, file):
    """
    path: path to mat file
    saves mat files as tsv in the same folder
    """
    mat = mat73.loadmat(f"{folder}/{file}")
    for k, v in mat.items():
        if not os.path.isfile(f"{folder}/{k}.tsv"):
            df = pd.DataFrame(v)
            df.to_csv(f"{folder}/{k}.tsv", sep="\t", index=False)
