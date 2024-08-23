package src;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class LouvainClusteringAlgorithm implements WeightedGraphClusteringAlgorithm {
	public List<List<Integer>> clusterNodes(int numVertices, List<WeightedEdge> graph) {
		List<Map<Integer,WeightedEdge>> adjacencyList = WeightedGraphClusteringAlgorithm.buildAdjacencyList(numVertices, graph);
		System.out.println("Built adj list with "+adjacencyList.size()+" nodes. Input vertices: "+numVertices);
		List<Map<Integer,WeightedEdge>> subgraphAdjacencyList = adjacencyList;
		List<List<Integer>> answer = new ArrayList<List<Integer>>();
		int [] clusterMemberships = new int [numVertices];
		
		for(int i=0;i<numVertices;i++) {
			clusterMemberships[i]=i;
			List<Integer> singleElemCluster = new ArrayList<Integer>();
			singleElemCluster.add(i);
			answer.add(singleElemCluster);
		}

		while (true) {
			System.out.print("Número de nodos del subgraph ------ ");
			System.out.println(subgraphAdjacencyList.size());
		//for (int r=0;r<1;r++) {
			int n = answer.size();
			int [] subgraphClusterMemberships = new int [n];
			List<Integer> indexes = new ArrayList<Integer>();
			for(int i=0;i<n;i++) {
				indexes.add(i);
				subgraphClusterMemberships[i]=i;
			}
			double totalWeight = calculateTotalWeight(subgraphAdjacencyList);
			boolean changed = false;
			while(true) {
				Collections.shuffle(indexes);
				System.out.println("Shuffled indexes. Nodes: "+indexes.size());
				boolean changedRound = false;
				for(int i:indexes) {
					int currentCluster = subgraphClusterMemberships[i];
					int newClusterI = calculateBestCluster(subgraphAdjacencyList,subgraphClusterMemberships,i, totalWeight);
					changedRound = changedRound || newClusterI!=currentCluster;
					subgraphClusterMemberships[i]=newClusterI;
				}
				if(!changedRound) break;
				changed = true;
			}

			if(!changed) break; 
			answer = updateClusters(clusterMemberships,subgraphClusterMemberships);

			int num_clusters = calculateNumberOfClusters(clusterMemberships);
			subgraphAdjacencyList = buildSubgraphAdjacencyList(adjacencyList, clusterMemberships, num_clusters);
				
		}
		return answer;
	}

	private int calculateNumberOfClusters(int [] array){ 
		Set<Integer> uniqueNumbers = new HashSet<>();
        for (int num : array) {
            uniqueNumbers.add(num);
        }
        int numberOfUniqueNumbers = uniqueNumbers.size();
		return numberOfUniqueNumbers;
	}

	private double calculateTotalWeight(List<Map<Integer, WeightedEdge>> subgraphAdjacencyList) {
		double w = 0;
		for(Map<Integer, WeightedEdge> edgesN:subgraphAdjacencyList) {
			for(WeightedEdge edge: edgesN.values()) {
				w+=edge.weight;
			}
		}
		return w;
	}

	private void printGraphWeights(List<Map<Integer,WeightedEdge>> subgraphAdjacencyList){
		for (int i=0;i<subgraphAdjacencyList.size();i++) {
			System.out.println("PARA EL NODO: " + i);

			Map<Integer,WeightedEdge> edgesI = subgraphAdjacencyList.get(i);
			for (Map.Entry<Integer, WeightedEdge> entry : edgesI.entrySet()) {
				WeightedEdge edge = entry.getValue();
				System.out.println("\n V1 " + edge.getV1());
				System.out.println(" V2: " + edge.getV2());
				System.out.println(" Weight: " + edge.getWeight());
			}
		}
	}

	private int calculateBestCluster(List<Map<Integer,WeightedEdge>> adjacencyList, int[] clusterMemberships, int i, double totalEdges) {
		Map<Integer,WeightedEdge> edgesI = adjacencyList.get(i);
		//if(i==0) System.out.println("Node: "+i+" edges: "+edgesI+" cluster: "+clusterMemberships[i]+ " totalEdges: "+totalEdges);
		int cI = clusterMemberships[i];
		Set<Integer> tried = new HashSet<Integer>();
		int bestCluster = cI;
		double bestModularityChange = 0;
		for(int j:edgesI.keySet()) {
			int cJ = clusterMemberships[j];
			if(cJ!=cI && !tried.contains(cJ) ) {
				double modularityChange = calculateModularityChange(adjacencyList,clusterMemberships,i,j, totalEdges);
				//if(i==0) System.out.println("Node1: "+i+" node2: "+j+" change: "+modularityChange);
				if(modularityChange>bestModularityChange) {
					bestCluster = cJ;
					bestModularityChange = modularityChange;
				}
				tried.add(cJ);
			}
		}
		if(bestModularityChange>0) System.out.println("Node "+i+" current: "+cI+" new: "+bestCluster+" mod change: "+bestModularityChange);
		return bestCluster;
	}

	
	private double calculateModularityChange(List<Map<Integer, WeightedEdge>> adjacencyList, int[] clusterMemberships, int i, int j, double totalEdges) {
		double currentM = calculateModularity(adjacencyList,clusterMemberships,clusterMemberships[i], totalEdges);
		currentM += calculateModularity(adjacencyList,clusterMemberships,clusterMemberships[j], totalEdges);
		int currentCluster = clusterMemberships[i];
		clusterMemberships[i] = clusterMemberships[j];
		double updatedM = calculateModularity(adjacencyList,clusterMemberships,currentCluster, totalEdges);
		updatedM += calculateModularity(adjacencyList,clusterMemberships,clusterMemberships[i], totalEdges);
		clusterMemberships[i] = currentCluster;
		return updatedM-currentM;
	}

	private double calculateModularity(List<Map<Integer, WeightedEdge>> adjacencyList, int[] clusterMemberships, int cluster, double totalEdges) {
		double sumIn = 0;
		double sumTot = 0;
		for(int i=0;i<clusterMemberships.length;i++) {
			if(clusterMemberships[i]!=cluster) continue;
			Map<Integer, WeightedEdge> edges = adjacencyList.get(i);
			for(WeightedEdge edge:edges.values()) {
				sumTot+=edge.getWeight();
				if(clusterMemberships[edge.v1]==clusterMemberships[edge.v2]) {
					sumIn+=edge.getWeight();
				}
			}
		}
		double m = sumIn;
		m/=(2.0*totalEdges);
		double m2 = sumTot*sumTot;
		m2/=(4.0*totalEdges*totalEdges);
		return m-m2;
	}

	private List<Map<Integer,WeightedEdge>> buildSubgraphAdjacencyList(List<Map<Integer,WeightedEdge>> adjacencyList, int [] clusterMemberships, int numClusters) {
		List<Map<Integer,WeightedEdge>> answer = new ArrayList<Map<Integer,WeightedEdge>>();
		for(int i=0;i<numClusters;i++) answer.add(new HashMap<Integer, WeightedEdge>());

		for(int v1=0;v1<clusterMemberships.length;v1++) {
			Map<Integer,WeightedEdge> edgesGraphV1 = adjacencyList.get(v1); // Vecinos de v1
			for(WeightedEdge edge :edgesGraphV1.values()) { // Para cada eje en los vecinos de v1
				int c1 = clusterMemberships[edge.getV1()];
				int c2 = clusterMemberships[edge.getV2()];
				WeightedEdge subgraphEdge = answer.get(c1).get(c2);
				if(subgraphEdge==null) {
					subgraphEdge = new WeightedEdge(c1, c2, edge.getWeight());
					WeightedGraphClusteringAlgorithm.addEdge(answer, subgraphEdge);
				} else
					subgraphEdge.addWeight(edge.getWeight());
			}
		}
		printGraphWeights(answer);

		return answer;
	}

	private List<List<Integer>> updateClusters(int[] clusterMemberships, int[] subgraphClusterMemberships) {
		for(int i=0;i<clusterMemberships.length;i++) {
			int oldCluster = clusterMemberships[i];
			clusterMemberships[i] = subgraphClusterMemberships[oldCluster];
		}
		List<List<Integer>>  answer = createClusters(clusterMemberships);
		//Reencode cluster ids
		for(int c=0;c<answer.size();c++) {
			for(int i:answer.get(c)) clusterMemberships[i] = c;
		}
		return answer;
	}

	private List<List<Integer>> createClusters(int[] clusterMemberships) {
		Map<Integer,List<Integer>> clusters = new HashMap<Integer, List<Integer>>();
		for(int i=0;i<clusterMemberships.length;i++) {
			List<Integer> members = clusters.computeIfAbsent(clusterMemberships[i], v->new ArrayList<Integer>());
			members.add(i);
		}
		List<List<Integer>> answer = new ArrayList<List<Integer>>();
		for(List<Integer> members:clusters.values()) answer.add(members);
		return answer;
	}
	public static void main(String[] args) throws Exception {
		int n=0;
		List<WeightedEdge> edges = new ArrayList<WeightedEdge>();
		try(FileReader reader = new FileReader(args[0]);
			BufferedReader in = new BufferedReader(reader)) {
			String line = in.readLine();
			
			while(line!=null) {
				String [] items = line.split(" ");
				int v1 = Integer.parseInt(items[0]);
				int v2 = Integer.parseInt(items[1]);
				double w = Double.parseDouble(items[2]);
				n = Math.max(n, v1+1);
				n = Math.max(n, v2+1);
				edges.add(new WeightedEdge(v1, v2, w));
				line = in.readLine();
			}
		}
		List<List<Integer>> clusters = (new LouvainClusteringAlgorithm()).clusterNodes(n, edges);
		for(List<Integer> cluster:clusters) System.out.println(cluster);
	}
}
