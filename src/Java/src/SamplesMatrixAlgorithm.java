package src;

public interface SamplesMatrixAlgorithm {
	/**
	 * Creates a square matrix for n samples given the information in the ScRNAMatrix.
	 * @param countsMatrix Matrix of single cell RNA-seq data
	 * @return double [][] nxn matrix. It can be either a similarity matrix or a distance matrix
	 */
	public double [][] generateSamplesMatrix(ScRNAMatrix countsMatrix);
}
