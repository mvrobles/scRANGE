import h5py
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score, silhouette_score)

def unsupervised_metrics(X, y_pred):
    # Realizamos PCA a 32 componentes
    X = PCA(n_components=32).fit_transform(X)

    # Evaluación final de resultados: métricas comparando con los clusters reales
    try:
        sil = np.round(silhouette_score(X, y_pred), 5)
        chs = np.round(calinski_harabasz_score(X, y_pred), 5)
        dbs = np.round(davies_bouldin_score(X, y_pred), 5)
    except:
        sil, chs, dbs = None, None, None

    return {'sil': sil, 'chs': chs, 'dbs': dbs}

def supervised_metrics(y_true, y_pred):
    acc = round(cluster_acc(y_true, y_pred),3)
    nmi = round(metrics.normalized_mutual_info_score(y_true, y_pred),3)
    ari = round(metrics.adjusted_rand_score(y_true, y_pred),3)

    return {'acc': acc, 'nmi': nmi, 'ari': ari}
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
