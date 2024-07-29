# imports
import os
import mat73
import pandas as pd
import polars as pl
from tqdm import tqdm

def parse_folder(path):
    """
    List all paths in directory.

    :param path: Path of base directory
    :type path: path-like
    :return: List of files and directories at path
    :rtype: list
    """    
    return os.listdir(path)

def mat_to_tsv(folder, file):
    """
    Saves mat file as tsv in the same folder.

    :param folder: path to folder with .mat files
    :type folder: path-like
    :param file:  name of .mat file
    :type file: str
    """
    mat = mat73.loadmat(f"{folder}/{file}")
    for k, v in mat.items():
        if not os.path.isfile(f"{folder}/{k}.tsv"):
            df = pd.DataFrame(v)
            df.to_csv(f"{folder}/{k}.tsv", sep="\t", index=False)

def mat_to_tsv_batch(folder:str):
    """
    Saves mat files as tsv in the same folder.

    :param folder: path to folder with .mat files
    :type folder: path-like
    """    
    for file in parse_folder(folder):
        if file.endswith(".mat"):
            mat_to_tsv(folder, file)

def convert_to_utf8(file):
    """
    Convert file from iso-8859-15 to UTF-8 encoding.

    :param file: Path to file
    :type file: path-like
    """    
    with open(file, 'r', encoding="iso-8859-15") as f:
        raw = f.read()
        splt = file.split(".")
        new_file = "".join(splt[:-1]) + "_utf8." + "".join(splt[-1])
        with open(new_file, "wb") as wf:
            wf.write(raw.encode("utf8"))

def remove_by_filename(directory_path, str:str):
    """
    Remove files at a dirctory by part of file name.

    :param directory_path: Path of base directory
    :type directory_path: path-like
    :param str: String to filter for
    :type str: str
    """    
    for file in os.listdir(directory_path):
        if str in file:
            os.remove(os.path.join(directory_path, file) )