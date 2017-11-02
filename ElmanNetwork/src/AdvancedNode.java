import java.util.LinkedList;
import java.util.Queue;

public abstract class AdvancedNode extends BasicNode{
	
	protected double net;
	protected double nodeDelta;
	protected Queue<Edge> leadingEdges;
	protected ActivationFunction activationFunction;
	
	public AdvancedNode(ActivationFunction activationFunction) {
		super();
		this.leadingEdges = new LinkedList<Edge>();
		this.activationFunction = activationFunction;
	}
	
	public Queue<Edge> getLeadingEdges() {return leadingEdges;}
	public void setLeadingEdges(Queue<Edge> leadingEdges) {this.leadingEdges = leadingEdges;}

	//just sets this.out and this.net
	public void feedForward() {
		this.net = 0;
		for(int i = 0; i < this.leadingEdges.size(); i += 1) {
			Edge next = this.leadingEdges.peek();
			this.leadingEdges.add(this.leadingEdges.poll());
			double toAdd = next.getWeight();
			toAdd *= next.getPrior().getOut();
			this.net += toAdd;
		}
		this.out = this.activationFunction.function(this.net);
	}
	
	//needs to be implemented
	public void setNodeDelta() {
		this.nodeDelta = 0;
	}
	
	public double getNodeDelta() {return this.nodeDelta;}
	
}
