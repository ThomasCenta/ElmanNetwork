
public abstract class BasicNode {

	protected double out;
	
	public static int largestId = 0;
	public int id;

	public BasicNode() {
		largestId+=1;
		this.id = largestId;
	}
	
	public double getOut() {
		return out;
	}

	public void setOut(double out) {
		this.out = out;
	}
	
}
