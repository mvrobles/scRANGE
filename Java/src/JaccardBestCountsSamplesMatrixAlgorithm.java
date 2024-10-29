package src;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class JaccardBestCountsSamplesMatrixAlgorithm implements SamplesMatrixAlgorithm {
	private int numThreads = 8;
	@Override
	public double[][] generateSamplesMatrix(ScRNAMatrix countsMatrix) {
		int n = countsMatrix.getCellIds().size();
		double [][] answer = new double [n][n];
		ThreadPoolExecutor pool = new ThreadPoolExecutor(numThreads, numThreads, 30, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
		for(int i=0;i<n;i++) {
			for(int j=i;j<n;j++) {
				final int c1 = i;
				final int c2 = j;
				pool.execute(()->calculateJaccard(countsMatrix,c1, c2, answer));
				//if(i%50==0 && answer[i][j]>0.1) System.out.println("Cell 1: "+i+" cell 2: "+j+" values cell2: "+valuesj.size()+" genes: "+geneIdxs.size()+" correl: "+answer[i][j] );
			}
		}
		pool.shutdown();
    	try {
			pool.awaitTermination(n*n*15, TimeUnit.SECONDS);
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		}
    	if(!pool.isShutdown()) {
			throw new RuntimeException("The ThreadPoolExecutor was not shutdown after an await Termination call");
		}
		return answer;
	}
	private void calculateJaccard(ScRNAMatrix countsMatrix, int i, int j, double[][] answer) {
		if(i==j) {
			answer[i][i] = 1;
			return;
		}
		Map<Integer,Double> valuesi = countsMatrix.getCountsCell(i, 5);
		if(valuesi==null) return;
		
		Map<Integer,Double> valuesj = countsMatrix.getCountsCell(j, 5);
		if(valuesj==null) return;
		
		Set<Integer> sInter = new HashSet<Integer>(valuesi.keySet());
		sInter.retainAll(valuesj.keySet());
		Set<Integer> sUnion = new HashSet<Integer>(valuesi.keySet());
		sUnion.addAll(valuesj.keySet());
		answer[i][j] = 1.0*sInter.size()/sUnion.size();
	}

}
