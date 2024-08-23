package src;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;

public class SCRNADataProcessor {
	private static final DecimalFormat ENGLISHFMT_PROBABILITIES = new DecimalFormat("0.0###",DecimalFormatSymbols.getInstance(Locale.ENGLISH));
	public static void main(String[] args) throws Exception {
		SCRNADataProcessor instance = new SCRNADataProcessor();
		//instance.processMatrixH5(args[0], args[1]);
		instance.processCellRangerMatrix(args[0], args[1]);
	}

	private void processCellRangerMatrix(String directory, String outPrefix) throws IOException {
		System.out.println("Loading cellRanger dataset");
		ScRNAMatrix matrix = loadCellRangerMatrix(directory);
		System.out.println("Loaded matrix");
		File theDir = new File(outPrefix);
		if (!theDir.exists()){
			theDir.mkdirs();
			System.out.println("Output folder created " + outPrefix);
		}
		processMatrix(matrix, outPrefix);
	}

	private ScRNAMatrix loadCellRangerMatrix(String directory) throws IOException {
		List<String> cellIds = loadIds(directory+"/barcodes.tsv.gz");
		System.out.println("Loaded "+cellIds.size()+" cell ids");
		List<String> geneIds = loadIds(directory+"/features.tsv.gz");
		System.out.println("Loaded "+geneIds.size()+" gene ids");
		List<List<Integer>> counts = new ArrayList<List<Integer>>();
		try (CellRangerMatrixFileReader reader = new CellRangerMatrixFileReader(directory+"/matrix.mtx.gz")) {
			Iterator<CellRangerCount> it = reader.iterator();
			while (it.hasNext()) {
				CellRangerCount count = it.next();
				List<Integer> countList = new ArrayList<Integer>();
				countList.add(count.getCellIdx());
				countList.add(count.getGeneIdx());
				countList.add((int) count.getCount());
				counts.add(countList);
			}
		}	
		return new ScRNAMatrix(cellIds, geneIds, counts);
	}

	private List<String> loadIds(String filename) throws IOException {
		List<String> ids = new ArrayList<String>();
		try(FileInputStream st1 = new FileInputStream(filename);
			ConcatGZIPInputStream st2 = new ConcatGZIPInputStream(st1);
			BufferedReader in = new BufferedReader(new InputStreamReader(st2))) {
			String line = in.readLine();
			while(line!=null) {
				int i = line.indexOf(" ");
				if(i>0) ids.add(line.substring(0,i));
				else ids.add(line);
				line = in.readLine();
			}
		}
		return ids;
	}

	// public void processMatrixH5(String filename, String outPrefix) throws IOException {
	// 	ScRNAMatrix matrix = (new H5Loader()).loadMatrix(filename);
	// 	processMatrix(matrix, outPrefix);
		
	// }

	public void processMatrix(ScRNAMatrix matrix, String outPrefix) throws IOException {
		long time0 = System.currentTimeMillis();
		matrix.filterGenes();
		matrix.filterCells();
		long time1 = System.currentTimeMillis();
		System.out.println("Calculating matrix. Loading time: "+((time1-time0)/1000));
		
		time0 = System.currentTimeMillis();
		//double[][] completeMatrix = matrix.toMatrix(matrix.getCellIds(), matrix.getGeneIds(), matrix.getCountsByCell());
		//System.out.println("Matriz completa");
		//completeMatrix = matrix.normalizeMatrix(completeMatrix);
		//System.out.println("Matriz normalizada");
		//matrix = new ScRNAMatrix(matrix.getCellIds(), matrix.getGeneIds(), normalizedMatrix); // Out of memory
		matrix.updateValues(matrix.normalizeMatrix(matrix.toMatrix()));

		time1 = System.currentTimeMillis();

		SamplesMatrixAlgorithm algMatrix = new PearsonCorrelationSamplesMatrixAlgorithm();
		//SamplesMatrixAlgorithm algMatrix = new JaccardBestCountsSamplesMatrixAlgorithm();
		double [][] correlations = algMatrix.generateSamplesMatrix(matrix);
		printCorrelationStats(correlations);
		long time2 = System.currentTimeMillis();
		System.out.println("Calculated matrix. Time: "+((time2-time1)/1000));
		saveSamplesMatrix(correlations, outPrefix+"_samplesMatrix.txt");
		long time3 = System.currentTimeMillis();
		System.out.println("Saved matrix. Time: "+((time3-time2)/1000));
		
		WeightedGraphGenerator graphGen = new KMSTWeightedGraphGenerator();
		//WeightedGraphGenerator graphGen = new KNNWeightedGraphGenerator();
		//WeightedGraphGenerator graphGen = new SimpleThresholdWeightedGraphGenerator();
		List<WeightedEdge> graph = graphGen.createWeightedGraph(correlations);
		saveGraph(graph, outPrefix+"_graph.txt");
		long time4 = System.currentTimeMillis();
		System.out.println("Calculated graph. Time: "+((time4-time3)/1000));
		long time5 = System.currentTimeMillis();
		List<String> cellIds = matrix.getCellIds();
		
		WeightedGraphClusteringAlgorithm algorithm = new LouvainClusteringAlgorithm();
		//WeightedGraphClusteringAlgorithm algorithm = new ConnectedComponentsClusteringAlgorithm();
		List<List<Integer>> clusters = algorithm.clusterNodes(correlations.length, graph);
		saveClusters(clusters, cellIds, outPrefix+"_clusters.txt");
		long time6 = System.currentTimeMillis();
		System.out.println("Calculated "+clusters.size()+" Clusters. Time: "+((time6-time5)/1000));
		System.out.println("Process finished. Total time: "+((time6-time0)/1000));
	}

	private void printCorrelationStats(double[][] correlations) {
		int [] counts = new int[21];
		for(int i=0;i<correlations.length;i++) {
			for(int j=i+1;j<correlations[i].length;j++) {
				int idx = (int)(correlations[i][j]*10+10);
				counts[idx]++;
			}
		}
		System.out.println("Correls dist");
		for(int i=0;i<counts.length;i++) {
			double val = 0.1*(i-10);
			System.out.println(""+val+" "+counts[i]);
		}
	}

	private void saveSamplesMatrix(double[][] samplesMatrix, String filename) throws IOException {	
		int n = samplesMatrix.length;
		try(PrintStream out = new PrintStream(filename)) {
			for(int i=0;i<n;i++) {
				out.print(""+i);
				for(int j=0;j<n;j++) {
					out.print("\t"+ENGLISHFMT_PROBABILITIES.format(samplesMatrix[i][j]));
				}
				out.println();
			}
		}
	}
	public void saveGraph(List<WeightedEdge> graph, String filename) throws IOException {
		try(PrintStream out = new PrintStream(filename)) {
			for(WeightedEdge edge: graph) {
				out.println(""+edge.getV1()+"\t"+edge.getV2()+"\t"+edge.getWeight());
			}
		}
	}

	private void saveClusters(List<List<Integer>> clusters, List<String> cellIds, String filename) throws IOException {
		int [] clusterMemberships = new int[cellIds.size()];
		for(int i=0;i<clusters.size();i++) {
			List<Integer> cluster = clusters.get(i);
			for(int j:cluster) {
				clusterMemberships[j]=i;
			}
		}
		try (PrintStream out = new PrintStream(filename)) {
			for(int i=0;i<clusterMemberships.length;i++) {
				out.println(""+i+"\t"+cellIds.get(i)+"\t"+clusterMemberships[i]);
			}
		}
	}
}
