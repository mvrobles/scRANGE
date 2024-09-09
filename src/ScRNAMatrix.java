package src;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class ScRNAMatrix {
	private List<String> cellIds;
	private List<Map<Integer,Double>> countsByCell= new ArrayList<Map<Integer,Double>>();
	private List<String> geneIds;
	private List<Map<Integer,Double>> countsByGene = new ArrayList<Map<Integer,Double>>();
	
	// Pruebas para normalización de matrices
	public static void main(String[] args) throws Exception {
		List<List<Integer>> counts = new ArrayList<>();

        // Leer el archivo y construir la lista de listas
        try (BufferedReader br = new BufferedReader(new FileReader(args[0]))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                List<Integer> countEntry = new ArrayList<>();
                for (String part : parts) {
                    countEntry.add(Integer.parseInt(part));
                }
                counts.add(countEntry);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

		List<String> cellIds = new ArrayList<>();
		List<String> geneIds = new ArrayList<>();

        for (int i = 0; i < 5; i++) {
            cellIds.add("celula" + i);
        }

        for (int i = 0; i < 3; i++) {
            geneIds.add("gen" + i);
        }

		ScRNAMatrix instance = new ScRNAMatrix(cellIds, geneIds, counts);

		instance.normalize();
		instance.printMatrix(instance.toMatrix());
	}

	public void printMatrix(double[][] matrixNorm) {
		for (int i = 0; i < matrixNorm.length; i++){
			System.out.println("\n");
			for (int j = 0; j < matrixNorm[i].length; j ++){
				System.out.print(matrixNorm[i][j] + " ");
			}
		}
	}

	public ScRNAMatrix(List<String> cellIds, List<String> geneIds, List<List<Integer>> counts) {
		super();
		this.cellIds = cellIds;
		this.geneIds = geneIds;
		initializeCounts(cellIds.size(), geneIds.size());
		for(List<Integer> countList:counts) {
			int cellId = countList.get(0);
			int geneId = countList.get(1);
			double count = (double) Math.min(Double.MAX_VALUE, countList.get(2));
			countsByCell.get(cellId).put(geneId, count);
			countsByGene.get(geneId).put(cellId, count);
		}
	}

	public ScRNAMatrix (double [][] fullMatrix) {
		initializeCounts(fullMatrix.length, fullMatrix[0].length);
		for(int i=0;i<fullMatrix.length;i++) {
			Map<Integer,Double> countsCell = countsByCell.get(i);
			for(int j=0;j<fullMatrix[i].length;j++) {
				Map<Integer,Double> countsGene = countsByGene.get(j);
				double value = fullMatrix[i][j];
				if(value==0) continue;
				if(value >Double.MAX_VALUE) value = Double.MAX_VALUE;
				countsCell.put(j,(double) value);
				countsGene.put(i,(double) value);
			}
		}
	}

	public ScRNAMatrix (List<String> cellIds, List<String> geneIds, double [][] fullMatrix) {
		super();
		this.cellIds = cellIds;
		this.geneIds = geneIds;
		initializeCounts(fullMatrix.length, fullMatrix[0].length);
		for(int i=0;i<fullMatrix.length;i++) {
			Map<Integer,Double> countsCell = countsByCell.get(i);
			for(int j=0;j<fullMatrix[i].length;j++) {
				Map<Integer,Double> countsGene = countsByGene.get(j);
				double value = fullMatrix[i][j];
				if(value==0) continue;
				if(value >Double.MAX_VALUE) value = Double.MAX_VALUE;
				countsCell.put(j,(double) value);
				countsGene.put(i,(double) value);
			}
		}
	}

	public void updateValues(double [][] fullMatrix){
		countsByCell= new ArrayList<Map<Integer,Double>>();
		countsByGene = new ArrayList<Map<Integer,Double>>();
		initializeCounts(fullMatrix.length, fullMatrix[0].length);
		for(int i=0;i<fullMatrix.length;i++) {
			Map<Integer,Double> countsCell = countsByCell.get(i);
			for(int j=0;j<fullMatrix[i].length;j++) {
				Map<Integer,Double> countsGene = countsByGene.get(j);
				double value = fullMatrix[i][j];
				if(value==0) continue;
				if(value >Double.MAX_VALUE) value = Double.MAX_VALUE;
				countsCell.put(j,(double) value);
				countsGene.put(i,(double) value);
			}
		}
	}

	public List<String> getCellIds() {
		return cellIds;
	}
	public List<String> getGeneIds() {
		return geneIds;
	}
	public List<Map<Integer, Double>> getCountsByCell(){
		return countsByCell;
	}

	public List<Map<Integer, Double>> getCountsByGene(){
		return countsByGene;
	}

	public Map<Integer,Double> getCountsCell(int cellIdx, int minValue) {
		Map<Integer,Double> answer = new TreeMap<Integer, Double>();
		for(Map.Entry<Integer,Double> entry:countsByCell.get(cellIdx).entrySet()) {
			if(entry.getValue()>=minValue) answer.put(entry.getKey(), entry.getValue());
		}
		return answer;
	}
	private void initializeCounts(int numGenes, int numCells) {
		for(int i=0;i<cellIds.size();i++) countsByCell.add(new TreeMap<Integer, Double>());
		for(int i=0;i<geneIds.size();i++) countsByGene.add(new TreeMap<Integer, Double>());	
	}
	
	public void filterGenes() {
		Set<Integer> idxsToRemove = new HashSet<Integer>();
		for(int j=0;j<countsByGene.size();j++) {
			Map<Integer,Double> countsGene= countsByGene.get(j);
			if(countsGene.size()<1) {
				idxsToRemove.add(j);
			}
			else if (countsGene.size()>10 & countsGene.size()<20){ 
				//System.out.println("GENE "+j);
				//System.out.println(countsGene);
			}
		}
		removeGenes(idxsToRemove);
		System.out.println("Remaining genes: "+countsByGene.size()+" removed: "+idxsToRemove.size());
	}

	private void removeGenes(Set<Integer> idxsToRemove) {
		List<String> geneIds2 = new ArrayList<String>();
		List<Map<Integer,Double>> countsByGene2 = new ArrayList<Map<Integer,Double>>();
		int j2 = 0;
		for(int j=0;j<countsByGene.size();j++) {
			boolean b = idxsToRemove.contains(j);
			if(b || j!=j2) {
				//System.out.println("Gene: "+j+" to remove: "+b+" new index: "+j2);
				Map<Integer,Double> countsGene= countsByGene.get(j);
				for(int i:countsGene.keySet()) {
					Map<Integer,Double> countsCell = countsByCell.get(i);
					double count = countsCell.remove(j);
					if(!b) countsCell.put(j2,count);
				}
			}
			if(!b) {
				geneIds2.add(geneIds.get(j));
				countsByGene2.add(countsByGene.get(j));
				j2++;
			}
		}
		geneIds = geneIds2;
		countsByGene = countsByGene2;
	}
	public void filterCells() {
		Set<Integer> idxsToRemove = new HashSet<Integer>();
		for(int i=0;i<countsByCell.size();i++) {
			Map<Integer,Double> countsCell = countsByCell.get(i);
			if(countsCell.size()<1) {
				System.out.println("Removing cell: "+i +" count: "+countsCell.size());
				idxsToRemove.add(i);
			}
		}
		removeCells(idxsToRemove);
		System.out.println("Remaining cells: "+countsByCell.size()+" removed: "+idxsToRemove.size());
	}
	private void removeCells(Set<Integer> idxsToRemove) {
		List<String> cellIds2 = new ArrayList<String>();
		List<Map<Integer,Double>> countsByCell2 = new ArrayList<Map<Integer,Double>>();
		int i2 = 0;
		for(int i=0;i<countsByCell.size();i++) {
			boolean b = idxsToRemove.contains(i); 
			if(b || i!=i2) {
				Map<Integer,Double> countsCell = countsByCell.get(i);
				for(int j:countsCell.keySet()) {
					Map<Integer,Double> countsGene = countsByGene.get(j);
					double count = countsGene.remove(i);
					if(!b) countsGene.put(i2,count);
				}
			}
			if(!b) {
				cellIds2.add(cellIds.get(i));
				countsByCell2.add(countsByCell.get(i));
				i2++;
			}
		}
		cellIds = cellIds2;
		countsByCell = countsByCell2;
	}

	public double[][] toMatrix() {
        int numCells = cellIds.size();
        int numGenes = geneIds.size();
        double[][] matrix = new double[numCells][numGenes];

        for (int i = 0; i < numCells; i++) {

            Map<Integer, Double> counts = countsByCell.get(i);
            for (Map.Entry<Integer, Double> entry : counts.entrySet()) {
				if (i == 0 && entry.getKey() == 2) {
					System.out.println("i: " + i + " key: " + entry.getKey() + " value: " + entry.getValue());
				}
                int geneIndex = entry.getKey();
                double count = entry.getValue();
                matrix[i][geneIndex] = count;
            }
        }
        return matrix;
    }

	private double[] getTotalSumByCell() {
		double[] totalSum = new double[countsByCell.size()];
		for (int i = 0; i < countsByCell.size(); i++){
			Map<Integer, Double> counts = countsByCell.get(i);
			for (Map.Entry<Integer, Double> entry : counts.entrySet()){
				totalSum[i] += entry.getValue();
			}
		}
		return totalSum;
	}

	public double computeMedian(double[] array){
		double[] copyArray = array.clone();
		Arrays.sort(copyArray);
		double median;
		if (copyArray.length % 2 == 0){
			median = (copyArray[copyArray.length/2] + copyArray[copyArray.length/2 - 1])/2;
		}
		else{
			median = copyArray[copyArray.length/2];
		}
		return median;
	}

	public void normalize(){
		double[] totalSum = getTotalSumByCell();
		double median = computeMedian(totalSum);

		for (int i = 0; i < countsByCell.size(); i++){
			Map<Integer, Double> counts = countsByCell.get(i);
			for (Map.Entry<Integer, Double> entry : counts.entrySet()){
				int geneIndex = entry.getKey();
				double count = entry.getValue();
				count = Math.log1p(count * (median / totalSum[i]));
				counts.put(geneIndex, count);
			}
		}
	}
}
