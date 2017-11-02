

public class OutputNode extends AdvancedNode {

	private double target;
	private ErrorFunction errorFunction;
	
	
	public OutputNode(ErrorFunction errorFunction, ActivationFunction activationFunction) {
		super(activationFunction);
		this.errorFunction = errorFunction;
	}

	@Override
	public void setNodeDelta() {
		this.nodeDelta = this.errorFunction.derivative(this.net, this.target, this.activationFunction);
	}
	
	public double getError() {
		return this.errorFunction.function(this.out, this.target);
	}
	
	public void setTarget(double target) {
		this.target = target;
	}
}
