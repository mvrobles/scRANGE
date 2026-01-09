import os
import argparse
from utils.read_data import get_paths, read_data_scexperiment
from kMST.kMST import run_kmst
from AE_GMM.NN_run_GMM import run_gmm

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
    else:
        raise Exception("Method not supported")

