import java.util.LinkedList;
import java.util.Queue;

public class HiddenNode extends AdvancedNode{

	private Queue<Edge> trailingEdges;
	
	public HiddenNode(ActivationFunction activationFunction) {
		super(activationFunction);
		this.trailingEdges = new LinkedList<Edge>();
	}
	
	public Queue<Edge> getTrailingEdges() {return trailingEdges;}

	@Override
	public void setNodeDelta() {
		double derivativeErrorWRTOut = 0;
		for(Edge edge: this.trailingEdges) {
			derivativeErrorWRTOut += edge.getWeight()*edge.getPost().getNodeDelta();
		}
		double derivativeOutWRTNet = this.activationFunction.derivative(this.net);
		this.nodeDelta = derivativeErrorWRTOut * derivativeOutWRTNet;
	}

}
