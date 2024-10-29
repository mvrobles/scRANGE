import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import multivariate_normal
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

from .layers import ClusteringLoss, DispAct, MeanAct, ZINBLoss

#import sys
#sys.path.append('../')

from .utils import *

torch.manual_seed(345)

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class scGMM(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, path, encodeLayer=[], decodeLayer=[], 
            activation="relu", sigma=1., alpha=1., ml_weight=1., cl_weight=1.):
        super(scGMM, self).__init__()
        
        # Inicialización de valores para el autoencoder
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.ml_weight = ml_weight
        self.cl_weight = cl_weight
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim]+decodeLayer, type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())
        
        # Inicialización de parámetros para el clustering
        self.mu = Parameter(torch.rand(n_clusters, z_dim, dtype=torch.float32))
        self.pi = Parameter(torch.rand(n_clusters, 1, dtype=torch.float32))
        self.diag_cov = Parameter(torch.ones(n_clusters, z_dim, dtype=torch.float32))

        # Funciones auxiliares: cálculo del ZINB loss, Softmax y Clustering Loss
        self.zinb_loss = ZINBLoss()#.cuda()
        self.softmax = nn.Softmax(dim=1)
        self.clustering_loss = ClusteringLoss()#.cuda()

        # Se guardan las covarianzas
        self.cov = torch.Tensor([np.identity(self.z_dim)]*self.n_clusters)

        # Directorio en donde se guardan los resultados
        self.path = path 

    
    def forward(self, x):
        x = x.to(torch.float32)
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)

        prob_matrix = self.find_probabilities(z0)
        return z0, _mean, _disp, _pi, prob_matrix
    
    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z, _, _, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def find_probabilities(self, Z):
        """
        Encuentra las probabilidades de cada punto a cada cluster a partir de las medias y las covarianzas.
        """
        #try: proba = torch.distributions.MultivariateNormal(self.mu.cuda(), self.cov.cuda()).log_prob(Z.cuda().unsqueeze(1))
        try: proba = torch.distributions.MultivariateNormal(self.mu, self.cov).log_prob(Z.unsqueeze(1))
        
        except: breakpoint()

        # Se resta el maximo número (en logaritmo)
        maximum = torch.max(proba,dim=1)[0]
        proba = proba - maximum[:,None]  

        # Se convierte a probabilidades
        proba = torch.exp(proba) 

        # Normalizamos
        proba = torch.div(proba,proba.sum(1).unsqueeze(-1))   
        
        # Multiplicamos por pi
        proba = torch.multiply(proba, nn.Softmax(dim=0)(self.pi).squeeze(1))

        # Completamos los 0s
        proba = torch.where(proba < 10**(-10), 10**(-10), proba.double())


        return proba

    def find_covariance(self, Z, mu, phi):
        """"
        Args:
            phi: Matriz (n_puntos x n_clusters) donde phi[i,k] representa la probabilidad de que el punto i esté en el cluster k.
            X: Matriz (n_puntos x d) con los puntos
            mu: Matriz (n_clusters x d) con las medias de cada cluster
        Returns:
            cov_mats: Lista (n_clusters) con una matriz de covarianza por cada cluster 
        """
        n_clus = self.n_clusters
        Z = Z.detach().numpy()
        mu = mu.detach().numpy()

        cov_mats = []
        for k in range(n_clus):
            nk = np.sum(phi[:,k])

            vects = []
            for i in range(self.z_dim):
                r =  np.matrix(Z[i,:] - mu[k,:])
                v = phi[i,k]*np.matmul( r.transpose(), r )
                vects.append(v)
            
            m = 1/nk*np.sum(vects, axis = 0)
            if nk == 0: m =  np.identity(self.z_dim)
            cov_mats.append(m)
        
        return cov_mats


    def pretrain_autoencoder(self, x, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400): # CAMBIAR POR 0.0001
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        
        loss_s = []
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch)#.cuda()
                x_raw_tensor = Variable(x_raw_batch)#.cuda()
                sf_tensor = Variable(sf_batch)#.cuda()
                _, mean_tensor, disp_tensor, pi_tensor, _ = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
                loss_s.append(loss.item())

    def fit(self, X, X_raw, sf, lr=0.1, batch_size=256, num_epochs=10, update_interval=1, tol=1e-4, y = None):
        '''X: tensor data'''
        #patience = 10
        #patience_counter = 0
        #best_loss = float('inf')
        
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        save_dir = self.path +'/'
        print("Clustering stage")
        X = torch.tensor(X)#.cuda()
        X_raw = torch.tensor(X_raw)#.cuda()
        sf = torch.tensor(sf)#.cuda()
        
        diag = torch.where(self.diag_cov.double() <= 0, 1/2100, self.diag_cov.double())
        x = [torch.diag(diag.detach()[i]) for i in range(self.n_clusters)]
        self.cov = torch.stack(x)#.cuda()

        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, n_init=20, random_state = 999)
        data = self.encodeBatch(X)

        self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        self.y_pred_last = self.y_pred

        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        
        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))

        clustering_metrics = {'ac': [], 'nmi': [], 'ari': []}
        losses = {'zinb': [], 'gmm': []}
        
        for epoch in range(num_epochs):
            print(f"---> Epoca {epoch}")

            if epoch%update_interval == 0:
                latent = self.encodeBatch(X)
                
                z = self.encodeBatch(X)

                diag = torch.where(self.diag_cov.double() <= 0, 1/2100, self.diag_cov.double())
                x = [torch.diag(diag.detach()[i]) for i in range(self.n_clusters)]
                self.cov = torch.stack(x)#.cuda() 

                distr = self.find_probabilities(z)
                self.y_pred = torch.argmax(distr.clone().detach() , dim=1).data.cpu().numpy()

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                    
                self.y_pred_last = self.y_pred
            
            cluster_loss_val = 0
            recon_loss_val = 0
            train_loss = 0
            # train 1 epoch for clustering loss
            for batch_idx in range(num_batch):

                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = sf[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]

                inputs = Variable(xbatch)
                rawinputs = Variable(xrawbatch)
                sfinputs = Variable(sfbatch)

                diag = torch.where(self.diag_cov.double() <= 0, 1/2100, self.diag_cov.double())
                self.cov = torch.stack([torch.diag(diag.detach()[i]) for i in range(self.n_clusters)])#.cuda()

                z, meanbatch, dispbatch, pibatch, prob_matrixbatch = self.forward(inputs)

                cluster_loss = self.clustering_loss(prob_matrixbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)
                
                loss = cluster_loss + recon_loss
                
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_value_(self.parameters(), 0)

                optimizer.step()

                cluster_loss_val += cluster_loss * len(inputs)
                recon_loss_val += recon_loss * len(inputs)
                train_loss = cluster_loss_val + recon_loss_val
                
            print("#Epoch %3d: Total: %.4f Clustering Loss: %.9f ZINB Loss: %.4f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))      

            losses['zinb'].append(recon_loss_val / num)
            losses['gmm'].append(cluster_loss_val / num)

        inputs = Variable(X)
        z, _, _, _, _ = self.forward(inputs)
        distr = self.find_probabilities(z).data.cpu().numpy()

        return self.y_pred, distr, self.mu, self.pi, self.diag_cov, z, epoch, clustering_metrics, losses
