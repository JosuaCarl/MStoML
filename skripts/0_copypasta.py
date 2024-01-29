# imports
import scipy.io
import argparse
import sklearn

# Argument access
parser = argparse.ArgumentParser(prog="ML4com.py",
                                 usage="ML4com.py"
                                       "-i <Path_to_n_files_or_folders or string> "
                                       "-o < PATH_FOR_PDF_OUTPUT> ")

# argument definition
parser.add_argument('-i', type=str, nargs="+", help="Location of input files. Can be a folder to take all files from that folder.")
parser.add_argument('-o', type=str, default=".", help="Location of output folder.")

# read out arguments
args = parser.parse_args()
infiles = args.i
outfile = args.o

### TRAINING

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Fit RandomForestClassifier
rfc.fit(X_train, y_train)
