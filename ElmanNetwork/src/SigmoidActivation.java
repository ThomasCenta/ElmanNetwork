
public class SigmoidActivation implements ActivationFunction{

	private double xElongation;
	private double xTranslation;
	
	
	public SigmoidActivation(double xElongation, double xTranslation) {
		this.xElongation = xElongation;
		this.xTranslation = xTranslation;
	}
	
	@Override
	public double derivative(double net) {
		return this.xElongation*this.function(net)*(1.0-this.function(net));
	}

	@Override
	public double function(double net) {
		double input = this.xElongation*net-this.xTranslation;
		return 1.0/(1+Math.exp(-1*input));
	}

}
