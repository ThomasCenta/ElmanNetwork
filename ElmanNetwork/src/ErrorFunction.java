
public interface ErrorFunction {

	//derivative w.r.t. net
	public double derivative(double net, double target, ActivationFunction activationFunction);
	
	//takes in a nodes output and its target output and calculates the error
	public double function(double output, double target);
	
}
