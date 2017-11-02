
public interface ActivationFunction {
	// derivative w.r.t. net
	public double derivative(double net);
	
	public double function(double net);
}
