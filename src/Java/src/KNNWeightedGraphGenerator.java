package src;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KNNWeightedGraphGenerator implements WeightedGraphGenerator {

	@Override
	public List<WeightedEdge> createWeightedGraph(double[][] samplesMatrix) {
		List<WeightedEdge> answer = new ArrayList<WeightedEdge>();
		int n = samplesMatrix.length;
		int k = (int) Math.log(n);
		for(int i=0;i<n;i++) {
			Map<Integer,Double> distances = selectBestEdges(samplesMatrix[i],i,k);
			for(Map.Entry<Integer, Double> entry:distances.entrySet()) {
				int j = entry.getKey();
				int w = (int) (1000*entry.getValue());
				answer.add(new WeightedEdge(i, j, w));
			}
		}
		 
		return answer;
	}

	private Map<Integer, Double> selectBestEdges(double[] values, int index, int k) {
		List<IndexWithValue> valuesList = new ArrayList<IndexWithValue>();
		for(int i=0;i<values.length;i++) {
			if(i!=index) valuesList.add(new IndexWithValue(i, (int) (10000*values[i])));
		}
		Collections.sort(valuesList,(k1,k2)->k2.value-k1.value);
		Map<Integer, Double> answer = new HashMap<Integer, Double>();
		for(int i=0;i<k;i++) {
			int j = valuesList.get(i).index;
			answer.put(j, values[j]);
		}
		return answer;
	}
}
class IndexWithValue {
	int index;
	int value;
	public IndexWithValue(int index, int value) {
		super();
		this.index = index;
		this.value = value;
	}
	
}
