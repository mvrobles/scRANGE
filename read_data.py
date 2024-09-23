import gzip
import numpy as np
import pandas as pd
from glob import glob
from scipy.io import mmread
from scipy.sparse import csc_matrix

def read_tsv_gz(file):
    """
    Reads a .tsv.gz file and returns a DataFrame.
    """
    with gzip.open(file, 'rt') as f:
        df = pd.read_csv(f, sep='\t', header=None)
    return df

def read_tsv(file):
    """
    Reads a .tsv file and returns a DataFrame.
    """
    with open(file, 'r') as f:
        df = pd.read_csv(f, sep='\t', header=None)
    return df

def read_mtx_gz(file) -> csc_matrix:
    """
    Reads a .mtx file and returns a sparse matrix.
    """
    with gzip.open(file, 'rt') as f:
        return mmread(f).tocsc().toarray().astype(np.float32)
    
def read_mtx(file) -> csc_matrix:
    """
    Reads a .mtx file and returns a sparse matrix.
    """
    with open(file, 'rt') as f:
        return mmread(f).tocsc().toarray().astype(np.float32)

def read_data_scexperiment(path_mtx, path_barcodes, path_features):
    if 'gz' in path_barcodes:
      barcodes = read_tsv_gz(path_barcodes)
    else:
      barcodes = read_tsv(path_barcodes)

    if 'gz'in path_features:
      genes = read_tsv_gz(path_features)
    else:
       genes = read_tsv(path_features)

    if 'gz' in path_mtx:
      x = read_mtx_gz(path_mtx).T
    else:
      x = read_mtx(path_mtx).T
    return barcodes, genes, x

def get_paths(path_input):
    path_mtx, path_barcodes, path_features = None, None, None
    for path in glob(path_input + '*'):
        if 'matrix.mtx' in path:
            path_mtx = path
        elif 'barcodes.tsv' in path:
            path_barcodes = path
        elif 'features.tsv' in path:
            path_features = path
        elif 'genes.tsv' in path:
            path_features = path


    assert path_mtx is not None, "Matrix file not found"
    assert path_features is not None, "Features file not found"
    assert path_barcodes is not None, "Barcodes file not found"

    return path_mtx, path_barcodes, path_features
