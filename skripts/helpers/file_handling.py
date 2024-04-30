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

def mat_to_tsv_batch(folder:str):
    for file in parse_folder(folder):
        if file.endswith(".mat"):
            mat_to_tsv(folder, file)

def convert_to_utf8(file):
    with open(file, 'r', encoding="iso-8859-15") as f:
        raw = f.read()
        splt = file.split(".")
        new_file = "".join(splt[:-1]) + "_utf8." + "".join(splt[-1])
        with open(new_file, "wb") as wf:
            wf.write(raw.encode("utf8"))

def remove_by_filename(directory_path, str:str):
    for file in os.listdir(directory_path):
        if str in file:
            os.remove(os.path.join(directory_path, file) )