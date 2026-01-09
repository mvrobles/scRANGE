package src;

public class CellRangerCount {
	private int cellIdx;
	private int geneIdx;
	private int count;
	public CellRangerCount(int cellIdx, int geneIdx, int count) {
		super();
		this.cellIdx = cellIdx;
		this.geneIdx = geneIdx;
		this.count = count;
	}
	public int getCellIdx() {
		return cellIdx;
	}
	public int getGeneIdx() {
		return geneIdx;
	}
	public int getCount() {
		return count;
	}
	
}
