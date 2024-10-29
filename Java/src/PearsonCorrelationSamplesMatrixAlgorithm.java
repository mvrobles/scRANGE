package src;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class PearsonCorrelationSamplesMatrixAlgorithm implements SamplesMatrixAlgorithm {

	private int numThreads = 4;
	@Override
	public double[][] generateSamplesMatrix(ScRNAMatrix countsMatrix) {
		int n = countsMatrix.getCellIds().size();
		System.out.println("Número de células " + n);
		double [][] answer = new double [n][n];
		ThreadPoolExecutor pool = new ThreadPoolExecutor(numThreads, numThreads, 30, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
		for(int i=0;i<n;i++) {
			for(int j=i;j<n;j++) {
				final int c1 = i;
				final int c2 = j;
				pool.execute(()->calculateCorrelation(countsMatrix,c1, c2, answer));
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
	private void calculateCorrelation(ScRNAMatrix countsMatrix, int i, int j, double [][] answer) {
		int debugI = -1;
		if(i==j) {
			answer[i][i] = 1;
			return;
		}
		Map<Integer,Double> valuesi = countsMatrix.getCountsCell(i, 0);
		if(valuesi==null) return;
		
		Map<Integer,Double> valuesj = countsMatrix.getCountsCell(j, 0);
		if(valuesj==null) return;
		Set<Integer> geneIdxs = new HashSet<Integer>(valuesi.keySet());
		geneIdxs.addAll(valuesj.keySet());
		answer[i][j] = calculateCorrelation(geneIdxs,valuesi,valuesj);
		answer[j][i] = answer[i][j];
		if(i==debugI && j==1) {
			printCounts(i,geneIdxs, valuesi);
			printCounts(j,geneIdxs, valuesj);
			System.out.println("Correl: "+answer[i][j]);
		}
		if(i%50==0 && j==answer.length-1) System.out.println("Calculated correlations for cell: "+i);
	}
	private double calculateCorrelation(Set<Integer> geneIdxs, Map<Integer, Double> valuesi, Map<Integer, Double> valuesj) {
		int n = geneIdxs.size();
		double si = calculateSum(valuesi.values());
		double sj = calculateSum(valuesj.values());
		double avgi = si/n;
		double avgj = sj/n;
		double sd1=0,sd2=0,sc=0;
		for(int g:geneIdxs) {
			double vi = valuesi.getOrDefault(g, (double) 0);
			double vj = valuesj.getOrDefault(g, (double) 0);
			
			double di = vi-avgi;
			double dj = vj-avgj;
			sd1 += (di*di);
			sd2 += (dj*dj);
			sc+= (di*dj);
		}
		sd1 = Math.sqrt(sd1);
		sd2 = Math.sqrt(sd2);
		return sc/(sd1*sd2);
	}
	private void printCounts(int cellIdx,Set<Integer> geneIdxs,  Map<Integer, Double> values) {
		System.out.println("Counts cell: "+cellIdx);
		for(int g:geneIdxs) {
			System.out.println("gene: "+g+" count: "+values.getOrDefault(g, (double) 0));
		}
	}
	
	private int calculateSum(Collection<Double> values) {
		int answer = 0;
		for(double v:values) answer+=v;
		return answer;
	}
}