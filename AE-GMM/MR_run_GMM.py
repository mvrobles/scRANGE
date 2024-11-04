import argparse
import os
import pickle
import warnings
from time import time
from datetime import datetime
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

from MR_GMM import scDCC

import sys
sys.path.append('../')

from preprocess import normalize, read_dataset
from utils import cluster_acc

warnings.filterwarnings('ignore')

def set_hyperparameters():
   # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("path_results")

    parser.add_argument('--label_cells', default=0.1, type=float)
    parser.add_argument('--label_cells_files', default='label_selected_cells_1.txt')
    parser.add_argument('--n_pairwise', default=0, type=int)
    parser.add_argument('--n_pairwise_error', default=0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=100, type=int) 
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--ml_weight', default=1., type=float,
                        help='coefficient of must-link loss')
    parser.add_argument('--cl_weight', default=1., type=float,
                        help='coefficient of cannot-link loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)

    args = parser.parse_args()

    return args

def read_data(path):
  data_mat = h5py.File(path)
  assert 'Y' in data_mat.keys() and 'X' in data_mat.keys()

  x = np.array(data_mat['X'], dtype = np.float64)
  if 'Y' in data_mat.keys():
    y = np.array(data_mat['Y'], dtype = np.float64)
  else: y = None
  data_mat.close()

  return x, y

def format_normalize(x, y):
    adata = sc.AnnData(x)
    if not y is None: adata.obs['Group'] = y

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    return adata

def create_train_model(args):
  # Create saving directory
  if not os.path.exists(args.path_results):
    os.makedirs(args.path_results)

  sd = 2.5
  
  # Model
  

    # Obtener la hora actual
  hora_actual = datetime.now()

  horas = hora_actual.hour
  minutos = hora_actual.minute
  segundos = hora_actual.second

  model = scDCC(input_dim=adata.n_vars, z_dim=32, n_clusters=n_clusters, 
              encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd,
              path = args.path_results, seed = horas + minutos + segundos).cuda()
  
  print(str(model))

  # Training
  t0 = time()
  if args.ae_weights is None:
      model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                              batch_size=args.batch_size, epochs=args.pretrain_epochs)
  else:
      if os.path.isfile(args.ae_weights):
          print("==> loading checkpoint '{}'".format(args.ae_weights))
          checkpoint = torch.load(args.ae_weights)
          model.load_state_dict(checkpoint['ae_state_dict'])
      else:
          print("==> no checkpoint found at '{}'".format(args.ae_weights))
          raise ValueError

  print('Pretraining time: %d seconds.' % int(time() - t0))

  return model

def second_training(args, model):
    t0 = time()
    
    # Second training: clustering loss + ZINB loss
    y_pred,  mu, pi, cov, z, epochs, clustering_metrics, losses = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors,  
                                    batch_size=args.batch_size,  num_epochs=args.maxiter,
                                    update_interval=args.update_interval, tol=args.tol, lr = 0.001, y = y)

    # Se guardan los resultados
    pd.DataFrame(z.cpu().detach().numpy()).to_csv(args.path_results + 'Z.csv', index = None)
    pd.DataFrame(mu.cpu().detach().numpy()).to_csv(args.path_results + 'Mu.csv', index = None)
    pd.DataFrame(pi.cpu().detach().numpy()).to_csv(args.path_results + 'Pi.csv', index = None)
    pd.DataFrame(cov.cpu().detach().numpy()).to_csv(args.path_results + 'DiagCov.csv', index = None)

    with open(args.path_results + '/prediccion.pickle', 'wb') as handle:
        pickle.dump(y_pred, handle)

    print('Time: %d seconds.' % int(time() - t0))

    return y_pred


def model_BIRCH(X: np.array, 
                n_clusters: int,
                threshold: list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 
                branching_factor: list =  [10, 50, 100, 150]
                ) -> tuple:
    """
    Trains a Birch model for the input data X by optimizing the hyperparameters threshold and branching factor.

    input:
    - X: array with data to cluster
    - n_clusters: number of clusters 
    - threshold: list of possible values of threshold for the Birch model
    - branching_factor: list of possible branching factor values for the Birch model

    output:
    - best_model: a Birch Model of sklearn that maximized the silhouette score for the data
    - params: tuple of 
    """
    # Hyperparameters to search
    param_grid = {
        'threshold': threshold,
        'branching_factor': branching_factor
    }

    max_sil = -2
    params = 0, 0 
    best_model = None 
    for t in param_grid['threshold']:
        for b in param_grid['branching_factor']:
            birch_model = Birch(n_clusters=n_clusters, threshold = t, branching_factor = b)
            birch_model.fit(X)

            labels = birch_model.predict(X)

            sil = silhouette_score(X, labels)
            if sil > max_sil:
                max_sil = sil
                params = t, b 
                best_model = birch_model

    return best_model, params, max_sil 

def unsupervised_metrics(X, y_pred):
    # Evaluación final de resultados: métricas comparando con los clusters reales
    sil = np.round(silhouette_score(X, y_pred), 5)
    chs = np.round(calinski_harabasz_score(X, y_pred), 5)
    dbs = np.round(davies_bouldin_score(X, y_pred), 5)
    print('Evaluating cells: SIL= %.4f, CHS= %.4f, DBS= %.4f' % (sil, chs, dbs))
    return {'sil': sil, 'chs': chs, 'dbs': dbs}

def supervised_metrics(y, y_pred):#, label_cell_indx):
    # Evaluación final de resultados: métricas comparando con los clusters reales
    assert y is not None 
    #print(len(y), label_cell_indx)
    #eval_cell_y_pred = np.delete(y_pred, label_cell_indx)
    #eval_cell_y = np.delete(y, label_cell_indx)
    
    #acc = np.round(cluster_acc(eval_cell_y, eval_cell_y_pred), 5)
    #nmi = np.round(metrics.normalized_mutual_info_score(eval_cell_y, eval_cell_y_pred), 5)
    #ari = np.round(metrics.adjusted_rand_score(eval_cell_y, eval_cell_y_pred), 5)
    
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

    #if not os.path.exists(args.label_cells_files):
    #    np.savetxt(args.label_cells_files, label_cell_indx, fmt="%i")
    return {'acc': acc, 'nmi': nmi, 'ari': ari}

if __name__ == "__main__":
    # Set hyperparameters
    args = set_hyperparameters()

    # Reading the data
    x, y = read_data(args.data_file)

    # processing of scRNA-seq read counts matrix
    adata = format_normalize(x, y)
    input_size = adata.n_vars

    n_clusters = int(n_clusters)

    # Model training
    model = create_train_model(args)   
    y_pred = second_training(args, model)

    print('----> Unsupervised metrics for GMM Autoencoder:')
    unsupervised_gmm = unsupervised_metrics(x, y_pred)

    # Supervised metrics
    if not y is None: 
       print('\n----> Supervised metrics for GMM Autoencoder:')
       supervised_gmm = supervised_metrics(y, y_pred)#, label_cell_indx)
    
    with open(args.path_results + 'metrics.pickle', 'wb') as file:
        pickle.dump({**unsupervised_gmm, **supervised_gmm}, file)

    # Birch
    try:
        z = pd.read_csv(args.path_results + 'Z.csv').values
        best_model, _, _ = model_BIRCH(X = z, n_clusters = n_clusters)
        y_pred = best_model.predict(z)
    
        print('\n----> Unsupervised metrics for GMM Autoencoder + Birch:')
        unsupervised_birch = unsupervised_metrics(z, y_pred)

        # Supervised metrics
        if not y is None: 
            print('\n----> Supervised metrics for GMM Autoencoder + Birch:')
            supervised_birch = supervised_metric(y, y_pred)#,label_cell_indx)

        with open(args.path_results + 'metrics_birch.pickle', 'wb') as file:
            pickle.dump({**unsupervised_birch, **supervised_birch}, file)
    except:
        print('Error Birch')

