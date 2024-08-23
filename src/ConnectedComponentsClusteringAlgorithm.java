package src;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

public class ConnectedComponentsClusteringAlgorithm implements WeightedGraphClusteringAlgorithm {

	@Override
	public List<List<Integer>> clusterNodes(int numVertices, List<WeightedEdge> edges) {
		List<List<Integer>> answer = new ArrayList<List<Integer>>();
		List<Map<Integer,WeightedEdge>> adjacencyList = WeightedGraphClusteringAlgorithm.buildAdjacencyList(numVertices, edges);
		boolean [] visited = new boolean[numVertices];
		for(int i=0;i<numVertices;i++) {
			if(visited[i]) continue;
			visited[i] = true;
			Queue<Integer> queue = new LinkedList<Integer>();
			queue.add(i);
			List<Integer> component = new ArrayList<Integer>();
			while(queue.size()>0) {
				int next = queue.poll();
				component.add(next);
				for(int j:adjacencyList.get(next).keySet()) {
					if(!visited[j]) {
						visited[j] = true;
						queue.add(j);
					}
				}
			}
			answer.add(component);
		}
		return answer;
	}

}
