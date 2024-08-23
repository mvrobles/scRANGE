package src;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public interface WeightedGraphClusteringAlgorithm {
	/**
	 * Builds clusters of nodes for the given graph
	 * @param numVertices Vertices of the graph. The nodes of the graph correspond to the numbers between 0 and numVertices-1
	 * @param edges List of weighted edges between the nodes. For each edge, the vertices are numbers between 0 and numVertices-1
	 * @return List<List<Integer>> Partition of the numbers between 0 and numVertices-1 representing the node clusters 
	 */
	public List<List<Integer>> clusterNodes(int numVertices, List<WeightedEdge> edges);
	
	public static List<Map<Integer,WeightedEdge>> buildAdjacencyList(int n, List<WeightedEdge> graph) {
		List<Map<Integer,WeightedEdge>> adjList = new ArrayList<Map<Integer,WeightedEdge>>();
		for(int i=0;i<n;i++) adjList.add(new HashMap<Integer,WeightedEdge>());
		for(WeightedEdge edge:graph) {
			addEdge(adjList, edge);
		}
		return adjList;


	}
	public static void addEdge(List<Map<Integer, WeightedEdge>> adjList, WeightedEdge edge) {
		adjList.get(edge.getV1()).put(edge.getV2(), edge);
		adjList.get(edge.getV2()).put(edge.getV1(),edge);
	}
}
