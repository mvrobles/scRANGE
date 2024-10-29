import argparse
import os
import pickle
import warnings
from time import time

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.cluster import Birch
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from .NN_GMM import scGMM

import sys
sys.path.append('../')

from .preprocess import normalize, read_dataset

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

    sd = 2.5

    # Model
    model = scGMM(input_dim=adata.n_vars, z_dim=32, n_clusters=n_clusters, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd,
                path = params['path_results'])#.cuda()

    print(str(model))

    # Training
    t0 = time()
    model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                            batch_size=params['batch_size'], epochs=params['pretrain_epochs'])

    print('Pretraining time: %d seconds.' % int(time() - t0))

    return model

def second_training(params, model, adata):
    t0 = time()
    
    # Second training: clustering loss + ZINB loss
    y_pred,  distr, mu, pi, cov, z, epochs, clustering_metrics, losses = model.fit(X=adata.X, X_raw=adata.raw.X, 
                                    sf=adata.obs.size_factors, batch_size=params['batch_size'],  num_epochs=params['maxiter'],
                                    update_interval=params['update_interval'], tol=params['tol'], lr = 0.001, y = None)

    # Se guardan los resultados
    pd.DataFrame(z.cpu().detach().numpy()).to_csv(params['path_results'] + 'Z.csv', index = None)
    pd.DataFrame(mu.cpu().detach().numpy()).to_csv(params['path_results']  + 'Mu.csv', index = None)
    pd.DataFrame(pi.cpu().detach().numpy()).to_csv(params['path_results']  + 'Pi.csv', index = None)
    pd.DataFrame(cov.cpu().detach().numpy()).to_csv(params['path_results'] + 'DiagCov.csv', index = None)

    print('Time: %d seconds.' % int(time() - t0))

    return distr, y_pred


def unsupervised_metrics(X, y_pred):
    # Evaluación final de resultados: métricas comparando con los clusters reales
    sil = np.round(silhouette_score(X, y_pred), 5)
    chs = np.round(calinski_harabasz_score(X, y_pred), 5)
    dbs = np.round(davies_bouldin_score(X, y_pred), 5)
    print('Evaluating cells: SIL= %.4f, CHS= %.4f, DBS= %.4f' % (sil, chs, dbs))

def run_GMM(X: np.array, 
            barcodes: pd.DataFrame,
            path_results: str, 
            n_clusters: int):
    params = set_hyperparameters()
    params['path_results'] = path_results

    # processing of scRNA-seq read counts matrix
    anndata_p = sc.AnnData(X)
    anndata_p.obs = barcodes

    # Normalize 
    anndata_p = normalize(anndata_p)
    barcodes = anndata_p.obs
    x = anndata_p.X 

    # Set k 
    n_clusters = int(n_clusters)

    # Model training
    model = create_train_model(params, anndata_p, n_clusters)   
    distr, y_pred1 = second_training(params, model, anndata_p)

    print('----> Unsupervised metrics for GMM Autoencoder:')
    unsupervised_metrics(x, y_pred1)

    # Guardar resultados
    barcodes['cluster'] = y_pred1
    barcodes.to_csv(path_results + 'gmm_clusters.csv', index = False)
    print(f'-----> Se guardó correctamente el csv {path_results}')

    for i in range(n_clusters):
        barcodes["prob_cluster" + str(i)] = distr[:,i]
        
    barcodes.to_csv(path_results + 'gmm_clusters_prob.csv', index = False)
    print(f'-----> Se guardó correctamente el csv con la distribución de probabilidad {path_results}')

    distr = distr / distr.sum(axis=1, keepdims=True)
    
    return barcodes, distr