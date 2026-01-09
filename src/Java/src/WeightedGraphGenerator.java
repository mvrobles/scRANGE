package src;
import java.util.List;

public interface WeightedGraphGenerator {
	/**
	 * Creates a weighted graph from the given samples matrix
	 * @param samplesMatrix a squared matrix. Numbers can represent similarities or distances
	 * @return List<WeightedEdge> Graph represented as a list of edges. Vertices of each edge must be numbers between 0 and samplesMatrix.length-1
	 */
	public List<WeightedEdge> createWeightedGraph(double [][] samplesMatrix);
}
