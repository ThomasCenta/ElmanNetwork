
public class Edge {
	
	private double weight;
	private BasicNode prior;
	private AdvancedNode post;
	private double previousVelocity;
	
	
	public Edge(BasicNode prior, AdvancedNode post, double weight) {
		this.prior = prior;
		this.weight = weight;
		this.post = post;
		this.previousVelocity = 0;
	}
	
	
	public double getWeight() {return this.weight;}
	public void setWeight(double weight) {this.weight = weight;}

	public BasicNode getPrior() {return prior;}
	public void setPrior(BasicNode prior) {this.prior = prior;}
	
	public AdvancedNode getPost() {return post;}
	public void setPost(AdvancedNode post) {this.post = post;}

	public void updateWeight(double learningRate, double momentum) {
		double gradient = post.getNodeDelta() * prior.getOut();
		double velocity = momentum*this.previousVelocity+learningRate*gradient;
		this.weight -= velocity;
		this.previousVelocity = velocity;
	}
	
	@Override
	public String toString() {
		return "Node "+this.prior.id+" connected to Node "+this.post.id+" with weight "+this.weight;
	}
}
