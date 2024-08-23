package src;
import java.util.ArrayList;
import java.util.List;

public class SimpleThresholdWeightedGraphGenerator implements WeightedGraphGenerator {

	private double threshold = 0.4;
	@Override
	public List<WeightedEdge> createWeightedGraph(double[][] samplesMatrix) {
		List<WeightedEdge> answer = new ArrayList<WeightedEdge>();
		for(int i=0;i<samplesMatrix.length;i++) {
			for(int j=i+1;j<samplesMatrix[i].length;j++) {
				if(samplesMatrix[i][j]<threshold) continue;
				double w = samplesMatrix[i][j]; // TODO MIRAR BIEN MELI
				answer.add(new WeightedEdge(i, j, w));
			}
		}
		return answer;
	}

}
