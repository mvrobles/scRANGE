import math
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

from AE_GMM.layers import ClusteringLoss, DispAct, MeanAct, ZINBLoss
from kMST.kMST import run_kmst
import shutil


def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class AEGMM(nn.Module):
    def __init__(
        self,
        input_dim,
        z_dim,
        path,
        encodeLayer=[],
        decodeLayer=[],
        activation="relu",
        sigma=1.0,
        alpha=1.0,
        ml_weight=1.0,
        cl_weight=1.0,
        n_clusters=None,
        seed=345,
    ):
        torch.manual_seed(seed)
        super(AEGMM, self).__init__()

        # Initialization of values for the autoencoder
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.ml_weight = ml_weight
        self.cl_weight = cl_weight
        self.encoder = buildNetwork(
            [input_dim] + encodeLayer, type="encode", activation=activation
        )
        self.decoder = buildNetwork(
            [z_dim] + decodeLayer, type="decode", activation=activation
        )
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(
            nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid()
        )

        # Clustering parameters will be initialized after pretraining
        self.mu = None
        self.pi = None
        self.diag_cov = None

        # Auxiliary functions: ZINB loss calculation, Softmax and Clustering Loss
        self.zinb_loss = ZINBLoss().cuda()
        self.softmax = nn.Softmax(dim=1)
        self.clustering_loss = ClusteringLoss().cuda()

        # Covariances will be initialized after pretraining
        self.cov = None

        # Directory where results are saved
        self.path = path

    def forward(self, x):
        x = x.to(torch.float32)
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)

        prob_matrix = self.find_probabilities(z0)
        return z0, _mean, _disp, _pi, prob_matrix

    def initialize_clustering_parameters(self, n_clusters):
        """
        Initialize clustering parameters (mu, pi, diag_cov, cov) after pretraining.

        Args:
            n_clusters: Number of clusters detected
        """
        self.n_clusters = n_clusters
        self.mu = Parameter(torch.rand(n_clusters, self.z_dim, dtype=torch.float32))
        self.pi = Parameter(torch.rand(n_clusters, 1, dtype=torch.float32))
        self.diag_cov = Parameter(
            torch.ones(n_clusters, self.z_dim, dtype=torch.float32)
        )

        # Initialize covariance matrices
        self.cov = torch.Tensor([np.identity(self.z_dim)] * n_clusters)

    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
            inputs = Variable(xbatch)
            z, _, _, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def find_probabilities(self, Z):
        """
        Find the probabilities of each point to each cluster based on means and covariances.
        """
        if self.mu is None or self.cov is None:
            raise RuntimeError(
                "Clustering parameters not initialized. Call initialize_clustering_parameters() first."
            )

        proba = torch.distributions.MultivariateNormal(
            self.mu.cuda(), self.cov.cuda()
        ).log_prob(Z.cuda().unsqueeze(1))

        # Subtract the max number
        maximum = torch.max(proba, dim=1)[0]
        proba = proba - maximum[:, None]

        # Convert to probabilities
        proba = torch.exp(proba)

        # Normalize
        proba = torch.div(proba, proba.sum(1).unsqueeze(-1))

        # Multiply by pi
        proba = torch.multiply(proba, nn.Softmax(dim=0)(self.pi).squeeze(1))

        # Complete zeros
        proba = torch.where(proba < 10 ** (-10), 10 ** (-10), proba.double())

        return proba

    def find_covariance(self, Z, mu, phi):
        """
        Args:
            phi: Matrix (n_points x n_clusters) where phi[i,k] represents the probability that point i is in cluster k.
            X: Matrix (n_points x d) with the points
            mu: Matrix (n_clusters x d) with the means of each cluster
        Returns:
            cov_mats: List (n_clusters) with one covariance matrix for each cluster
        """
        n_clus = self.n_clusters
        Z = Z.detach().numpy()
        mu = mu.detach().numpy()

        cov_mats = []
        for k in range(n_clus):
            nk = np.sum(phi[:, k])

            vects = []
            for i in range(self.z_dim):
                r = np.matrix(Z[i, :] - mu[k, :])
                v = phi[i, k] * np.matmul(r.transpose(), r)
                vects.append(v)

            m = 1 / nk * np.sum(vects, axis=0)
            if nk == 0:
                m = np.identity(self.z_dim)
            cov_mats.append(m)

        return cov_mats

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, "FTcheckpoint_%d.pth.tar" % index)
        torch.save(state, newfilename)

    def pretrain_autoencoder(
        self, x, X_raw, size_factor, batch_size=256, lr=0.0001, epochs=400
    ):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(
            torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True
        )

        loss_s = []
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).cuda()
                x_raw_tensor = Variable(x_raw_batch).cuda()
                sf_tensor = Variable(sf_batch).cuda()
                _, mean_tensor, disp_tensor, pi_tensor, _ = self.forward(x_tensor)
                loss = self.zinb_loss(
                    x=x_raw_tensor,
                    mean=mean_tensor,
                    disp=disp_tensor,
                    pi=pi_tensor,
                    scale_factor=sf_tensor,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(
                    "Pretrain epoch [{}/{}], ZINB loss:{:.4f}".format(
                        batch_idx + 1, epoch + 1, loss.item()
                    )
                )
                loss_s.append(loss.item())

            with open(self.path + "/pretrain_loss.pickle", "wb") as handle:
                pickle.dump(loss_s, handle)

        # Save the pretrained model
        f = open(self.path + f"pretrained_model_with_{epoch}_epochs.pickle", "wb")
        pickle.dump(self, f)
        f.close

    def estimate_n_clusters(self, X_latent):
        if self.n_clusters is not None:
            return self.n_clusters
        print("Estimating clusters...")
        
        # Create temp directory if it doesn't exist
        temp_path = self.path + "/temp/"
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        run_kmst(
            X=X_latent,
            barcodes=["c" + str(i) for i in range(len(X_latent))],
            path_results=temp_path,
            filter="mean-variance",
        )

        clusters = pd.read_csv(temp_path + "clusters.csv")["cluster"]
        n_clusters = len(set(clusters))

        print(f"Number of clusters: {n_clusters}")

        self.n_clusters = n_clusters
        shutil.rmtree(temp_path)
        return n_clusters

    def fit(
        self,
        X,
        X_raw,
        sf,
        lr=0.1,
        batch_size=256,
        num_epochs=10,
        update_interval=1,
        tol=1e-4,
        y=None,
    ):
        """X: tensor data"""
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        save_dir = self.path + "/"
        print("Clustering stage")
        X = torch.tensor(X).cuda()
        X_raw = torch.tensor(X_raw).cuda()
        sf = torch.tensor(sf).cuda()

        diag = torch.where(
            self.diag_cov.double() <= 0, 1 / 2100, self.diag_cov.double()
        )
        x = [torch.diag(diag.detach()[i]) for i in range(self.n_clusters)]
        self.cov = torch.stack(x).cuda()

        optimizer = optim.Adadelta(
            filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=0.95
        )

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, n_init=20, random_state=999)
        data = self.encodeBatch(X)

        self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        self.y_pred_last = self.y_pred

        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        losses = {"zinb": [], "gmm": []}

        for epoch in range(num_epochs):
            print(f"---> Epoch {epoch}")

            if epoch % update_interval == 0:
                latent = self.encodeBatch(X)

                z = self.encodeBatch(X)

                diag = torch.where(
                    self.diag_cov.double() <= 0, 1 / 2100, self.diag_cov.double()
                )
                x = [torch.diag(diag.detach()[i]) for i in range(self.n_clusters)]
                self.cov = torch.stack(x).cuda()

                distr = self.find_probabilities(z)
                self.y_pred = (
                    torch.argmax(distr.clone().detach(), dim=1).data.cpu().numpy()
                )

                # Save current model
                if epoch % 50 == 0:
                    self.save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "state_dict": self.state_dict(),
                            "mu": self.mu,
                            "y_pred": self.y_pred,
                            "z": z,
                            "pi": self.pi,
                            "cov": self.cov,
                        },
                        epoch + 1,
                        filename=save_dir,
                    )

                self.y_pred_last = self.y_pred

            cluster_loss_val = 0
            recon_loss_val = 0
            train_loss = 0
            # Train 1 epoch for clustering loss
            for batch_idx in range(num_batch):
                xbatch = X[
                    batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)
                ]
                xrawbatch = X_raw[
                    batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)
                ]
                sfbatch = sf[
                    batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)
                ]

                inputs = Variable(xbatch)
                rawinputs = Variable(xrawbatch)
                sfinputs = Variable(sfbatch)

                diag = torch.where(
                    self.diag_cov.double() <= 0, 1 / 2100, self.diag_cov.double()
                )
                self.cov = torch.stack(
                    [torch.diag(diag.detach()[i]) for i in range(self.n_clusters)]
                ).cuda()

                z, meanbatch, dispbatch, pibatch, prob_matrixbatch = self.forward(
                    inputs
                )

                cluster_loss = self.clustering_loss(prob_matrixbatch)
                recon_loss = self.zinb_loss(
                    rawinputs, meanbatch, dispbatch, pibatch, sfinputs
                )

                loss = cluster_loss + recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cluster_loss_val += cluster_loss * len(inputs)
                recon_loss_val += recon_loss * len(inputs)
                train_loss = cluster_loss_val + recon_loss_val

            print(
                "#Epoch %3d: Total: %.4f Clustering Loss: %.9f ZINB Loss: %.4f"
                % (
                    epoch + 1,
                    train_loss / num,
                    cluster_loss_val / num,
                    recon_loss_val / num,
                )
            )

            losses["zinb"].append(recon_loss_val / num)
            losses["gmm"].append(cluster_loss_val / num)

            with open(self.path + "/losses.pickle", "wb") as handle:
                pickle.dump(losses, handle)

            with open(self.path + "/decoder_info.pickle", "wb") as handle:
                inputs = Variable(X)
                z, meanbatch, dispbatch, pibatch, prob_matrixbatch = self.forward(
                    inputs
                )
                results_encoder = {"mean": meanbatch, "sigma": dispbatch, "pi": pibatch}
                pickle.dump(results_encoder, handle)

        inputs = Variable(X)
        z, _, _, _, _ = self.forward(inputs)
        distr = self.find_probabilities(z).data.cpu().numpy()

        return self.y_pred, distr, self.mu, self.pi, self.diag_cov, z, epoch, losses
