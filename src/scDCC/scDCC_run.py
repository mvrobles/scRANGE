from time import time
import os

import h5py
import numpy as np
import scanpy as sc
from scipy.optimize import linear_sum_assignment
import torch
from scDCC.scDCC import scDCC
from scDCC.preprocess import read_dataset, normalize

def read_data(path):
  data_mat = h5py.File(path)
  assert 'Y' in data_mat.keys() and 'X' in data_mat.keys()

  x = np.array(data_mat['X'], dtype = np.float64)
  if 'Y' in data_mat.keys():
    y = np.array(data_mat['Y'], dtype = np.float64)
  else: y = None
  data_mat.close()

  return x, y
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def run_scdcc(X, barcodes, path_results, n_clusters, 
          batch_size, maxiter, pretrain_epochs, gamma, update_interval,
          tol, ae_weights, ae_weight_file):
    adata = sc.AnnData(X)

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])
    
    sd = 2.5

    model = scDCC(input_dim=adata.n_vars, z_dim=32, n_clusters=n_clusters, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd, gamma=gamma).cuda()
    
    print(str(model))

    t0 = time()
    if ae_weights is None:
        model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                batch_size=batch_size, epochs=pretrain_epochs, ae_weights=path_results + ae_weight_file)
    else:
        if os.path.isfile(ae_weights):
            print("==> loading checkpoint '{}'".format(ae_weights))
            checkpoint = torch.load(ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(ae_weights))
            raise ValueError
    
    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, batch_size=batch_size, num_epochs=maxiter, 
                ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
                update_interval=update_interval, tol=tol, save_dir=path_results)
    print('Total time: %d seconds.' % int(time() - t0))

    barcodes['cluster'] = np.array(y_pred)
    barcodes.to_csv(path_results + 'scdcc_clusters.csv', index = False)
    print(f'-----> Correctly saved {path_results}')

    print('Total time: %d seconds.' % int(time() - t0))

