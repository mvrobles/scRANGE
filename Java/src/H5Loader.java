package src;
// import java.nio.file.Paths;
// import java.util.Map;

// import io.jhdf.HdfFile;
// import io.jhdf.api.Attribute;
// import io.jhdf.api.Dataset;
// import io.jhdf.api.Group;
// import io.jhdf.api.Node;

// public class H5Loader {
// 	public ScRNAMatrix loadMatrix(String filename) {
// 		ScRNAMatrix answer;
// 		try (HdfFile hdfFile = new HdfFile(Paths.get(filename))) {
// 			int [][] matrix = recursivePrintGroup(hdfFile, 0);
// 			answer = new ScRNAMatrix(matrix);
// 		}
// 		return answer;
// 	}
// 	private int[][] recursivePrintGroup(Group group, int level) {
// 		int [][] answer=null;
// 		level++;
// 		for (Node node : group) {
// 			System.out.println(node.getName()+" "+level); //NOSONAR - sout in example
// 			if (node instanceof Group) {
// 				recursivePrintGroup((Group) node, level);
// 			} else {
// 				System.out.println("Type: "+node.getType()+" Path: "+node.getPath());
// 				Map<String,Attribute> atts = node.getAttributes();
// 				for(Map.Entry<String, Attribute> entry:atts.entrySet() ) System.out.println("Next attribute: "+entry.getKey()+" value: " +entry.getValue().getName()+" "+entry.getValue().getSize());
// 				Dataset dataset = (Dataset)node;
// 				System.out.println("Dataset data class: "+dataset.getData().getClass());
// 				if (node.getName().equals("X")) {
// 					try {
// 						double [][] data = (double[][]) dataset.getData();
// 						answer = roundMatrix(data);
// 					} catch (Exception e) {
// 						answer = (int[][]) dataset.getData();
// 					}
// 					System.out.println("Rows: "+answer.length+" cols: "+answer[0].length+" first: "+answer[0][0]);
// 				} else {
// 					try {
// 						double [] data = (double[]) dataset.getData();
// 						System.out.println("values: "+data.length+" first: "+data[0]);
// 						//for(int i=0;i<data.length;i++) if(i%100==0) System.out.println("Next: "+i+" value: "+data[i]);
// 					} catch (Exception e) {
// 						// TODO Auto-generated catch block
// 						e.printStackTrace();
// 					}	
// 				}
				
// 			}
// 		}
// 		return answer;
// 	}
// 	private int[][] roundMatrix(double[][] data) {
// 		int [][] answer = new int [data.length][data[0].length];
// 		for(int i=0;i<data.length;i++) {
// 			for(int j=0;j<data[0].length;j++) answer[i][j] = (int) Math.round(data[i][j]);
// 		}
// 		return answer;
// 	}
// }
