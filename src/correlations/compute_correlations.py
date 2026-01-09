import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 
import pandas as pd 
import numpy as np
import argparse
import os 

import sys
sys.path.append('../')

from utils.read_data import get_paths, read_data_scexperiment

def process_hpa():
    """
    Reads the Human Protein Atlas single-cell RNA data, pivots the table to have gene names as rows and 
    a multi-index of (Tissue, Cell type, Cluster) as columns, with read counts as values.

    Returns:
        pandas.DataFrame: Pivoted DataFrame with gene names as index and (Tissue, Cell type, Cluster) as columns.
    """
    hpa_type_tissue ='../Data/HumanProteinAtlas/rna_single_cell_type_tissue.tsv'
    df = pd.read_csv(hpa_type_tissue, sep = '\t')
    df = df.pivot_table(
        index=['Gene name'], 
        columns=['Tissue', 'Cell type', 'Cluster'], 
        values='Read count').T
    return df

def select_genes(hpa, sc):
    """
    Selects and returns the subset of genes (columns) that are present in both the hpa and sc DataFrames.

    Args:
        hpa (pd.DataFrame): DataFrame containing gene expression data (genes as columns).
        sc (pd.DataFrame): DataFrame containing gene expression data (genes as columns).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Filtered hpa and sc DataFrames containing only the shared genes.
    """
    sc = sc.loc[:,~sc.columns.duplicated()]
    intersection_genes = list(set(hpa.columns).intersection(set(sc.columns)))
    hpa = hpa[intersection_genes]
    sc = sc[intersection_genes]

    return hpa, sc

def calculate_correlation(array1, array2):
    """
    Calculates the correlation coefficients between two arrays.

    Args:
        array1 (np.ndarray): The first input array. Should be a 1D or 2D NumPy array.
        array2 (np.ndarray): The second input array. Should have the same number of columns as array1 if 2D.

    Returns
        np.ndarray: An array containing the correlation coefficients between the rows of array1 and array2.
    """
    combined = np.vstack((array1, array2))
    correlation_matrix = np.corrcoef(combined)
    correlation_array = correlation_matrix[0:array1.shape[0], array1.shape[0]:]
    return correlation_array

def assign_results(corr, hpa, sc):
    """
    Assigns the most correlated type and tissue for each entry in the input correlation matrix.

    Args:
        corr (np.ndarray): A 2D correlation matrix where rows correspond to types (from `hpa`) and columns correspond to entries (from `sc`).
        hpa (pd.DataFrame): A DataFrame where each entry is a tuple or list containing (tissue, type) information.
        sc (pd.DataFrame): A DataFrame whose index corresponds to the columns of `corr`.

    Returns:
        pd.DataFrame: A DataFrame with the same index as `sc`, containing columns:
            - 'tissue': The tissue name corresponding to the highest correlation for each entry.
            - 'type': The type name corresponding to the highest correlation for each entry.
            - 'max_corr': The maximum correlation value for each entry.
    """
    idxmax = corr.argmax(axis = 0)

    max_corr = corr.max(axis = 0)

    results = pd.DataFrame(columns = ["type", "max_corr"])
    results['type'] = idxmax
    results['max_corr'] = max_corr

    results['type'] = results['type'].apply(lambda x: hpa.index[x])
    results['tissue'] = results['type'].apply(lambda x: x[0])
    results['type'] = results['type'].apply(lambda x: x[1])

    results = results[['tissue', 'type', 'max_corr']]
    results.index = sc.index
    return results

def assign_cell_type_chunk(sc: pd.DataFrame):
    """
    Assigns cell types to single-cell data by computing correlations with reference data.

    This function processes Human Protein Atlas (HPA) data, selects common genes between
    the HPA and single-cell (sc) DataFrames, computes the correlation matrix between
    the reference and single-cell data, and assigns cell type results based on the
    correlation scores.

    Args:
        sc (pd.DataFrame): Single-cell gene expression DataFrame. Rows represent genes,
            columns represent individual cells.

    Returns:
        pd.DataFrame: DataFrame containing the assigned cell types or correlation results
            for each cell in the input single-cell data.
    """
    hpa = process_hpa()
    hpa, sc = select_genes(hpa, sc)
    corr = calculate_correlation(hpa.values, sc.values)
    results = assign_results(corr, hpa, sc)

    return results

def assign_cell_type(barcodes, genes, X):
    """
    Assigns cell types to a set of single-cell data based on gene expression.
    This function processes the input data in chunks if the number of barcodes exceeds 2000,
    to avoid memory issues. 

    Args:
        barcodes (numpy.array): List of cell barcode identifiers.
        genes (numpy.array): List of gene names corresponding to columns in X.
        X (numpy.ndarray): 2D array of gene expression values, where rows
            correspond to cells (barcodes) and columns correspond to genes.

    Returns:
        pandas.DataFrame: DataFrame containing the assigned cell types for each barcode.
    """
    if len(barcodes) > 2000:
        results = pd.DataFrame()
        for i in tqdm(range(0, len(barcodes), 2000)):
            barcodes_i = barcodes[i:i+2000]
            X_i = X[i:i+2000,:]
            data_i = pd.DataFrame(X_i, columns = genes, index = barcodes_i)
            results = pd.concat([results, assign_cell_type_chunk(data_i)])
    
    else:
        data = pd.DataFrame(X, columns = genes, index = barcodes)
        results = assign_cell_type_chunk(data)

    return results

def plot_results(results, path_results):
    """
    Plots the results of correlation analysis and saves the figure.

    This function creates a figure with two subplots:
    1. A histogram showing the distribution of the maximum correlation values (`results.max_corr`).
    2. A horizontal bar plot showing the distribution of assigned tissues (`results.tissue.value_counts()`).

    The resulting figure is saved as 'plot_result.png' in the specified output directory.

    Args:
        results (pd.DataFrame): DataFrame containing at least the columns 'max_corr' (float) and 'tissue' (categorical or string).
        path_results (str): Path to the directory where the resulting plot image will be saved.

    Returns:
        None
    """
    fig, axes = plt.subplots(ncols = 2, dpi = 300, figsize = (10,3.5))

    plt.subplots_adjust(wspace = 0.5)
    sns.histplot(results.max_corr, kde=True, ax = axes[0]).set(
        title = "Max correlation distribution",
        ylabel = "Correlation",
        xlabel = "",
        xlim = (0,1)
    )
    b= sns.barplot(results.tissue.value_counts(), orient='h')
    b.set(
        title = "Assign tissue distribution",
        ylabel = "",
        xlabel = "Barcodes",
    )

    for container in b.containers:
        b.bar_label(container, label_type='edge', padding=3, fontsize=8)

    plt.savefig(path_results + 'plot_result.png', dpi=600, bbox_inches='tight')

def run_correlations(path_results, genes, barcodes, X):
    genes = genes[genes.columns[1]].values
    barcodes = barcodes[barcodes.columns[0]].values
    
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    results = assign_cell_type(barcodes, genes, X)
    results.to_csv(path_results+ 'results.csv')

    plot_results(results, path_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input")
    parser.add_argument("path_results")
    args = parser.parse_args()

    path_mtx, path_barcodes, path_features = get_paths(args.path_input)
    barcodes, genes, X = read_data_scexperiment(path_mtx, path_barcodes, path_features)

    genes = genes[genes.columns[1]].values
    barcodes = barcodes[barcodes.columns[0]].values
    
    if not os.path.exists(args.path_results):
        os.makedirs(args.path_results)

    results = assign_cell_type(barcodes, genes, X)
    results.to_csv(args.path_results+ 'results.csv')

    plot_results(results, args.path_results)