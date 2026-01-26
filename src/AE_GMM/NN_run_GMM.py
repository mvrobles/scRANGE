import argparse
import os
import sys
import warnings
from time import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from AE_GMM.NN_GMM import AEGMM
from AE_GMM.preprocess import normalize

sys.path.append('../')
from utils.read_data import get_paths, read_data_scexperiment

warnings.filterwarnings('ignore')

def set_hyperparameters():
    parameters = {
        'label_cells': 0.1,
        'label_cells_files': 'label_selected_cells_1.txt',
        'n_pairwise': 0,
        'n_pairwise_error': 0,
        'batch_size': 256,
        'maxiter': 100,
        'pretrain_epochs': 300,
        'gamma': 1.,
        'ml_weight': 1.,
        'cl_weight': 1.,
        'update_interval': 1,
        'tol': 0.001,
    }

    return parameters

def create_train_model(params, adata, n_clusters):
    # Create saving directory
    if not os.path.exists(params['path_results']):
        os.makedirs(params['path_results'])

    # Model
    if n_clusters == 0:
        n_clusters = None 

    model = AEGMM(input_dim=adata.n_vars, z_dim=32, n_clusters=n_clusters, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=2.5,
                path = params['path_results']).cuda()

    print(str(model))

    # Training
    t0 = time()
    model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                            batch_size=params['batch_size'], epochs=params['pretrain_epochs'])

    print('Pretraining time: %d seconds.' % int(time() - t0))

    return model

def estimate_number_clusters(model, adata):
    """Estimate the number of clusters using the latent representation."""
    # Convert data to tensor and move to GPU
    X_tensor = torch.tensor(adata.X, dtype=torch.float32).cuda()
    
    # Get latent representation (no gradient needed)
    with torch.no_grad():
        X_latent, _, _, _, _ = model.forward(X_tensor)
    
    # Estimate number of clusters
    n_clusters = model.estimate_n_clusters(X_latent)
    
    # Initialize clustering parameters
    model.initialize_clustering_parameters(n_clusters)
    
    return n_clusters


def second_training(params, model, adata):
    t0 = time()
    
    # Second training: clustering loss + ZINB loss
    y_pred,  distr, mu, pi, cov, z, epochs, losses = model.fit(X=adata.X, X_raw=adata.raw.X, 
                                    sf=adata.obs.size_factors, batch_size=params['batch_size'],  num_epochs=params['maxiter'],
                                    update_interval=params['update_interval'], tol=params['tol'], lr = 0.001, y = None)

    # Save results
    pd.DataFrame(z.cpu().detach().numpy()).to_csv(params['path_results'] + 'Z.csv', index = None)

    print('Time: %d seconds.' % int(time() - t0))

    return distr, y_pred

def unsupervised_metrics(X, y_pred):
    # Final evaluation of results: metrics comparing with real clusters
    sil = np.round(silhouette_score(X, y_pred), 5)
    chs = np.round(calinski_harabasz_score(X, y_pred), 5)
    dbs = np.round(davies_bouldin_score(X, y_pred), 5)
    print('Evaluating cells: SIL= %.4f, CHS= %.4f, DBS= %.4f' % (sil, chs, dbs))

def run_gmm(X: np.array, 
            barcodes: pd.DataFrame,
            path_results: str, 
            n_clusters: int):
    params = set_hyperparameters()
    params['path_results'] = path_results

    # Processing of scRNA-seq read counts matrix
    anndata_p = sc.AnnData(X)
    anndata_p.obs = barcodes

    # Normalize data
    anndata_p = normalize(anndata_p)
    barcodes = anndata_p.obs
    x = anndata_p.X 

    # Set number of clusters
    n_clusters = int(n_clusters)

    # Model training: pretrain autoencoder
    model = create_train_model(params, anndata_p, n_clusters)
    
    # Estimate or initialize clustering parameters
    if n_clusters == 0:
        n_clusters = estimate_number_clusters(model, anndata_p)
    else:
        model.initialize_clustering_parameters(n_clusters)
    
    # Second training: clustering loss + ZINB loss
    distr, y_pred1 = second_training(params, model, anndata_p)

    print('----> Unsupervised metrics for GMM Autoencoder:')
    unsupervised_metrics(x, y_pred1)

    # Save results
    barcodes['cluster'] = y_pred1
    barcodes.to_csv(path_results + 'gmm_clusters.csv', index = False)
    print(f'Results saved to {path_results}')

    for i in range(n_clusters):
        barcodes["prob_cluster" + str(i)] = distr[:,i]
        
    barcodes.to_csv(path_results + 'gmm_clusters_prob.csv', index = False)
    print(f'Probability distribution saved to {path_results}')

    distr = distr / distr.sum(axis=1, keepdims=True)
    
    return barcodes, distr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input")
    parser.add_argument("path_results")
    parser.add_argument("n_clusters")

    args = parser.parse_args()

    path_mtx, path_barcodes, path_features = get_paths(args.path_input)
    barcodes, genes, X = read_data_scexperiment(path_mtx, path_barcodes, path_features)

    if not os.path.exists(args.path_results):
        os.makedirs(args.path_results)

    run_gmm(X = X, 
            barcodes = barcodes,
            path_results = args.path_results,
            n_clusters = args.n_clusters)
