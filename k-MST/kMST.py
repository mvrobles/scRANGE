import argparse
import pickle
import warnings

from glob import glob
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.optimize import curve_fit
from tqdm import tqdm
import os 

from read_data import *

warnings.filterwarnings('ignore')

def normalize(adata, filter_min_counts=True, logtrans_input=True):
    """
    Normalize the AnnData object by filtering genes and cells with minimum counts,
    normalizing per cell, and optionally applying log transformation.

    Parameters:
    adata (AnnData): The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    filter_min_counts (bool, optional): If True, filter genes and cells with at least one count. Default is True.
    logtrans_input (bool, optional): If True, apply log transformation to the data. Default is True.

    Returns:
    AnnData: The normalized AnnData object.

    Prints:
    The percentage of zero values in the data matrix.
    """
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    adata.raw = adata
    sc.pp.normalize_per_cell(adata)

    if logtrans_input:
        sc.pp.log1p(adata) 

    zero_percentage = np.where(adata.X==0,True,False).sum()/(adata.X.shape[0]*adata.X.shape[1])
    print('Cero percentage', zero_percentage)

    return adata

def filter_genes_variance(X):
    """
    Filters the genes in the input matrix X based on their variance.

    This function calculates the variance of each column (gene) in the input matrix X,
    sorts the genes by their variance in descending order, and selects the top 5000
    genes with the highest variance. The filtered matrix containing only these top
    5000 genes is then returned.

    Parameters:
    X (numpy.ndarray): A 2D array where rows represent samples and columns represent genes.

    Returns:
    numpy.ndarray: A 2D array containing only the top 5000 genes with the highest variance.
    """
    varianzas_columnas = np.var(X, axis=0)
    indices_mayor_varianza = np.argsort(varianzas_columnas)[::-1][:5000]
    X_filtered = X[:, indices_mayor_varianza]

    return X_filtered

def filter_genes_mean_variance(X):
    medias = np.mean(X, axis = 0)
    varianzas = np.var(X, axis = 0)

    def curve_func(x, a, n, b):
        return a * x / (x**n + b)

    popt, _ = curve_fit(curve_func, medias, varianzas)
    a,n,b = popt

    expected_variance = a * medias / (medias**n + b)
    observed_expected = varianzas - expected_variance
    indices_mayor_var = np.argsort(observed_expected)[::-1][:5000]
    X_filtered = X[:, indices_mayor_var]

    return X_filtered

def paint_matrix(correlaciones, output_path, name):
    plt.figure()
    sns.heatmap(correlaciones).set(
        title = 'Correlaciones entre células')
    plt.savefig(output_path + name + '.png')
    plt.close()

def get_correlations(X):
    correlaciones = np.corrcoef(X)
    return correlaciones

def save_matrix(correlaciones, output_path, name):
    df = pd.DataFrame(correlaciones)
    df.to_csv(output_path + name + '.csv', index=False)


def create_kmst(distance_matrix, inverse = True, k = None, threshold = 1e-5):
    if k is None:
        N = np.log(len(distance_matrix))
        k = int(np.floor(N))
    
    print(f'k = {k}')
    grafo = generate_initial_graph(distance_matrix, inverse, threshold)

    print(f'---> Number of edges: {grafo.number_of_edges()}')

    mst_antes = None
    for iter in tqdm(range(k)):
        mst_new = nx.minimum_spanning_tree(grafo)

        edges_to_remove = list(mst_new.edges)
        grafo.remove_edges_from(edges_to_remove)

        if mst_antes is None:
            mst_antes =mst_new.copy()
        else:
            mst_new.add_edges_from(list(mst_antes.edges(data=True)))
            mst_antes = mst_new.copy()

    return mst_antes 

def generate_initial_graph(distance_matrix, inverse, threshold):
    grafo = nx.Graph()
    nodos = range(len(distance_matrix))

    # Crear nodo inicial
    grafo.add_nodes_from(nodos)

    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix[i])):
            peso = distance_matrix[i][j]
            if peso > threshold:
                # para MST necesito el inverso de las correlaciones
                if inverse:
                    grafo.add_edge(i, j, weight=1-peso)
                else:
                    grafo.add_edge(i, j, weight=peso)
    return grafo

def louvain(grafo):
    particiones = nx.community.louvain_communities(grafo, seed=123)

    diccionario = {}

    for i, conjunto in enumerate(particiones):
        for elemento in conjunto:
            diccionario[elemento] = i

    num_nodos = grafo.number_of_nodes()
    clusters = np.full(num_nodos, -1, dtype=int)

    for nodo, comunidad in diccionario.items():
        clusters[nodo] = comunidad

    return clusters

def run_kmst(X: np.array, 
             barcodes: pd.DataFrame, 
             path_results: str, 
             filter: str) -> pd.DataFrame:
    # Converting to AnnData
    anndata_p = sc.AnnData(X)
    anndata_p.obs = barcodes

    # Normalize 
    anndata_p = normalize(anndata_p)
    barcodes = anndata_p.obs
    
    X = anndata_p.X

    # Gene selection
    if filter == 'mean-variance':
        X = filter_genes_mean_variance(X)
    elif filter == 'variance':
        X = filter_genes_variance(X)
    
    print(f"Dimensions: {anndata_p.X.shape}")

    # Correlations
    correlaciones = get_correlations(X)

    # KMST
    kmst = create_kmst(distance_matrix = correlaciones, 
                       inverse = True, 
                       threshold = 0)

    # Clustering
    clusters = louvain(kmst)
    barcodes['cluster'] = clusters
    barcodes.to_csv(path_results + 'clusters.csv', index = False)

    return barcodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input")
    parser.add_argument("path_results")
    parser.add_argument("filter")
    args = parser.parse_args()

    path_mtx, path_barcodes, path_features = get_paths(args.path_input)
    barcodes, genes, X = read_data_scexperiment(path_mtx, path_barcodes, path_features)

    if not os.path.exists(args.path_results):
        os.makedirs(args.path_results)

    run_kmst(X = X, 
             barcodes = barcodes, 
             path_results = args.path_results,
             filter = args.filter)