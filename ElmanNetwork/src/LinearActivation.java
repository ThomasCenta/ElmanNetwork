
public class LinearActivation implements ActivationFunction{

	@Override
	public double derivative(double net) {
		return 1;
	}

	@Override
	public double function(double net) {
		return net;
	}

}
