import os
import argparse
from utils.read_data import get_paths, read_data_scexperiment
from kMST.kMST import run_kmst
from AE_GMM.NN_run_GMM import run_gmm
from scDCC.scDCC_run import run_scdcc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="method",
        required=True
    )
    # ---------- kmst -----------
    kmst_parser = subparsers.add_parser("kmst")
    kmst_parser.add_argument("path_input")
    kmst_parser.add_argument("path_results")
    kmst_parser.add_argument("--filter", type = str, required=True)

    # ---------- ae-gmm -----------
    aegmm_parser = subparsers.add_parser("ae-gmm")
    aegmm_parser.add_argument("path_input")
    aegmm_parser.add_argument("path_results")
    aegmm_parser.add_argument("--n_clusters", type = int, required=True)
    
    # ---------- scdcc ----------
    scdcc_parser = subparsers.add_parser(
        "scdcc",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    scdcc_parser.add_argument("path_input")
    scdcc_parser.add_argument("path_results")

    scdcc_parser.add_argument("--n_clusters", required=True, type=int)
    scdcc_parser.add_argument("--n_pairwise", default=0, type=int)
    scdcc_parser.add_argument("--batch_size", default=256, type=int)
    scdcc_parser.add_argument("--maxiter", default=2000, type=int)
    scdcc_parser.add_argument("--pretrain_epochs", default=300, type=int)
    scdcc_parser.add_argument("--gamma", default=1.0, type=float,
                              help="coefficient of clustering loss")
    scdcc_parser.add_argument("--update_interval", default=1, type=int)
    scdcc_parser.add_argument("--tol", default=0.001, type=float)
    scdcc_parser.add_argument("--ae_weights", default=None)

    scdcc_parser.add_argument("--ae_weight_file", default="AE_weights_p0_1.pth.tar")
    scdcc_parser.add_argument("--latent_z", default="latent_p0_1.txt")

    args = parser.parse_args()
    
    path_mtx, path_barcodes, path_features = get_paths(args.path_input)
    barcodes, genes, X = read_data_scexperiment(path_mtx, path_barcodes, path_features)

    if not os.path.exists(args.path_results):
        os.makedirs(args.path_results)

    if args.method == 'kmst':
        run_kmst(X = X, 
             barcodes = barcodes, 
             path_results = args.path_results,
             filter = args.filter)
    elif args.method == 'ae-gmm':
        run_gmm(X = X,
            barcodes = barcodes,
            path_results = args.path_results,
            n_clusters = args.n_clusters)
    elif args.method == 'scdcc':
        run_scdcc(
            X=X,
            barcodes=barcodes,
            path_results=args.path_results,
            n_clusters=args.n_clusters,
            batch_size=args.batch_size,
            maxiter=args.maxiter,
            pretrain_epochs=args.pretrain_epochs,
            gamma=args.gamma,
            update_interval=args.update_interval,
            tol=args.tol,
            ae_weights=args.ae_weights,
            ae_weight_file=args.ae_weight_file,
        )
    else:
        raise Exception("Method not supported")

