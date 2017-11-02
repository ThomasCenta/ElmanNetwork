
public class ArcTanActivation implements ActivationFunction{

	@Override
	public double derivative(double net) {
		return 1/(net*net+1);
	}

	@Override
	public double function(double net) {
		return Math.atan(net);
	}

}
