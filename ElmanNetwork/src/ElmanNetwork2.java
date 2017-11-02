import java.util.Random;

public class ElmanNetwork2 {

	
	private double getWeightInitialization() {
		double toReturn = rand.nextDouble()*4+1;
		if(rand.nextBoolean()){
			toReturn *= -1;
		}
		return toReturn;
	}
	
	
	private static Random rand = new Random();
	/* hidden x (input + context + bias) in that order
	 *  note that the i position determines hidden nodes, j determines the (input + context + bias) nodes
	 */
	private double[][] inputHiddenWeights;
	private double[][] previousInputHiddenWeights;
	/*
	 * output x hidden
	 */
	private double[][] hiddenOutputWeights;
	private double[][] previousHiddenOutputWeights;
	private double[] contextValues;
	private ActivationFunction hiddenActivation;
	private ActivationFunction outputActivation;
	private ErrorFunction errorFunction;
	private int numInputs;
	private int numHiddenNodes;
	private int numOutputs;
	
	public ElmanNetwork2(int numInputs, int numHiddenNodes, int numOutputs, ActivationFunction hiddenActivationFunction, ActivationFunction outputActivationFunction, ErrorFunction errorFunction) {
		this.inputHiddenWeights = new double[numHiddenNodes][numInputs+numHiddenNodes+1];
		for(int i = 0; i < numHiddenNodes; i += 1) {
			for(int j = 0; j < (numInputs+numHiddenNodes+1); j += 1) {
				this.inputHiddenWeights[i][j] = getWeightInitialization();
			}
		}
		
		this.hiddenOutputWeights = new double[numOutputs][numHiddenNodes];
		for(int i = 0; i < numOutputs; i += 1) {
			for(int j = 0; j < numHiddenNodes; j += 1) {
				this.hiddenOutputWeights[i][j] = getWeightInitialization();
			}
		}
		
		this.hiddenActivation = hiddenActivationFunction;
		this.outputActivation = outputActivationFunction;
		this.errorFunction = errorFunction;
		this.contextValues = new double[numHiddenNodes];
		this.numInputs = numInputs;
		this.numHiddenNodes = numHiddenNodes;
		this.numOutputs = numOutputs;
	}
	
	//this does matrix x vector
	private static double[] matrixMultiply(double[][] matrix, double[] vector) {
		assert matrix[0].length == vector.length;
		
		double[] toReturn = new double[matrix.length];
		for(int i = 0; i < toReturn.length; i += 1) {
			double sum = 0;
			for(int j = 0; j < vector.length; j += 1) {
				sum += matrix[i][j] * vector[j];
			}
			toReturn[i] = sum;
		}
		return toReturn;
	}
	
	//this does matrixT x vector
	private static double[] matrixTransposeMultiply(double[][] matrix, double[] vector) {
		assert matrix.length == vector.length;
		
		double[] toReturn = new double[matrix[0].length];
		for(int j = 0; j < toReturn.length; j += 1) {
			double sum = 0;
			for(int i = 0; i < vector.length; i += 1) {
				sum += matrix[i][j] * vector[i];
			}
			toReturn[j] = sum;
		}
		return toReturn;
	}
		
	
	
	/*
	 *  sets the context node values to the outputs of the hidden values
	 *  returns the net values for hidden nodes
	 */
	private double[] processInput(double[] input) {
		assert input.length == this.numInputs;
		
		double[] inputVector = new double[this.numInputs+this.numHiddenNodes+1];
		int iterator = 0;
		for(int i = 0; i < input.length; i += 1) {
			inputVector[iterator] = input[i];
			iterator++;
		}
		for(int i = 0; i < this.contextValues.length; i += 1) {
			inputVector[iterator] = this.contextValues[i];
			iterator++;
		}
		inputVector[iterator] = 1; // bias value
		
		double[] newHiddenNets = matrixMultiply(this.inputHiddenWeights, inputVector);
		for(int i = 0; i < this.contextValues.length; i += 1) {
			double output = this.hiddenActivation.function(newHiddenNets[i]);
			this.contextValues[i] = output;
		}
		return newHiddenNets;
	}
	
	private double[] getOutputNets(double[] input) {
		double[] hiddenNets = this.processInput(input);
		double[] hiddenOuts = new double[this.numHiddenNodes];
		for(int i = 0; i < this.numHiddenNodes; i += 1) {
			hiddenOuts[i] = this.hiddenActivation.function(hiddenNets[i]);
		}
		return matrixMultiply(this.hiddenOutputWeights, hiddenOuts);
	}
	
	//this returns the nets for the outputs
	public double[] getOutput(double[] input) {
		double[] outputNets = this.getOutputNets(input);
		double[] outputs = new double[this.numOutputs];
		for(int i = 0; i < this.numOutputs; i += 1) {
			outputs[i] = this.outputActivation.function(outputNets[i]);
		}
		return outputs;
	}
	
	public void setContextValues(double[] values) {
		assert values.length == this.numHiddenNodes;
		
		for(int i = 0; i < values.length; i++) {
			this.contextValues[i] = values[i];
		}
	}
	
	private static void copy(double[] copyFrom, double[] copyTo) {
		for(int i = 0; i < copyTo.length; i += 1) {
			copyTo[i] = copyFrom[i];
		}
	}
	
	private double[] hiddenNodeDeltasFromOutputNodeDeltas(double[] outputNodeDeltas, double[] hiddenNodeNets) {
		double[] hiddenNodeDeltas = matrixTransposeMultiply(this.hiddenOutputWeights, outputNodeDeltas);
		for(int i = 0; i < hiddenNodeDeltas.length; i += 1) {
			hiddenNodeDeltas[i] *= this.hiddenActivation.derivative(hiddenNodeNets[i]);
		}
		return hiddenNodeDeltas;
	}
	
	private double[] contextNodeDeltasFromHiddenNodeDeltas(double[] hiddenNodeDeltas) {
		double[] contextNodeDeltas = new double[hiddenNodeDeltas.length];
		for(int i = 0; i < contextNodeDeltas.length; i += 1) {
			for(int j = 0; j < hiddenNodeDeltas.length; j += 1) {
				contextNodeDeltas[i] += this.inputHiddenWeights[j][this.numInputs+i]*hiddenNodeDeltas[j];
			}
		}
		return contextNodeDeltas;
	}
	
	private double[] hiddenNodeDeltasFromContextNodeDeltas(double[] contextNodeDeltas, double[] hiddenNodeNets) {
		double[] hiddenNodeDeltas = new double[this.numHiddenNodes];
		for(int i = 0; i < hiddenNodeDeltas.length; i += 1) {
				hiddenNodeDeltas[i] = this.hiddenActivation.derivative(hiddenNodeNets[i])*contextNodeDeltas[i];
		}
		return hiddenNodeDeltas;
	}

	
	
	
	private double[] hiddenOutsFromNets(double[] hiddenNets) {
		double[] toReturn = new double[hiddenNets.length];
		for(int i = 0; i < hiddenNets.length; i += 1) {
			toReturn[i] = this.hiddenActivation.function(hiddenNets[i]);
		}
		return toReturn;
	}
	
	//weightMatrix assumed to be post x prior
	private static void updateColumnWeights(double[] priorNodeOutputs, double[] postNodeDeltas, double[][] currentWeightMatrix, double[][] previousEdgeChanges, double learningRate, double momentum){
		for(int i = 0; i < postNodeDeltas.length; i += 1) {
			for(int j = 0; j < priorNodeOutputs.length; j += 1) {
				double gradient = priorNodeOutputs[j]*postNodeDeltas[i];
				double edgeChange = momentum*previousEdgeChanges[i][j] - gradient*learningRate;
				previousEdgeChanges[i][j] = edgeChange;
				currentWeightMatrix[i][j] += edgeChange;
			}
		}
	}
	
	private void updateWeights(double[][] inputs, double[][] hiddenNetValues, double[] outputNodeDeltas, double[][] hiddenNodeDeltas, double[][] contextNodeDeltas, double learningRate, double momentum) {
		updateColumnWeights(hiddenOutsFromNets(hiddenNetValues[hiddenNetValues.length-1]), outputNodeDeltas, this.hiddenOutputWeights, this.previousHiddenOutputWeights, learningRate, momentum);
		int numUnfolds = inputs.length;
		for(int i_u = 0; i_u < numUnfolds; i_u += 1) {
			double[] inputOuts = new double[this.numInputs+this.numHiddenNodes+1];
			int index = 0;
			for(int i = 0 ; i < inputs[i_u].length; i += 1) {
				inputOuts[index] = inputs[i_u][i];
				index += 1;
			}
			//this is where to pick up from next time.
			/*
			 * 
			 * 
			 * 
			 * 
			 * Making sure you saw
			 * 
			 * 
			 * 
			 */
			
		}
	}
	
	//updates the weights, returns the error
	public double train(double[][] input, double[] expectedOutput, double learningRate) {
		assert input.length > 0;
		assert input[0].length == this.numInputs;
		assert this.numOutputs == expectedOutput.length;
		
		//calculate net and output values for all nodes
		int numUnfolds = input.length;
		double[][] hiddenNetValues = new double[numUnfolds][this.numHiddenNodes];
		double[][] contextNetValues = new double[numUnfolds][this.numHiddenNodes];
		for(int i = 0; i < numUnfolds; i += 1) {
			copy(this.contextValues, contextNetValues[i]);
			double[] hiddenNets = this.processInput(input[i]);
			copy(hiddenNets, hiddenNetValues[i]);
		}
		double[] hiddenOuts = hiddenOutsFromNets(hiddenNetValues[numUnfolds-1]);
		double[] outputNets = matrixMultiply(this.hiddenOutputWeights, hiddenOuts);
		double[] outputs = new double[this.numOutputs];
		for(int i = 0; i < this.numOutputs; i += 1) {
			outputs[i] = this.outputActivation.function(outputNets[i]);
		}
		
		//time for the node deltas
		double[] outputNodeDeltas = new double[this.numOutputs];
		double[][] hiddenNodeDeltas = new double[numUnfolds][this.numHiddenNodes];
		double[][] contextNodeDeltas = new double[numUnfolds][this.numHiddenNodes];
		
		double sum = 0.0;
		for(int i = 0; i < this.numOutputs; i += 1) {
			outputNodeDeltas[i] = this.errorFunction.derivative(outputNets[i], expectedOutput[i], this.outputActivation);
			sum += this.errorFunction.function(outputNets[i], expectedOutput[i]);
		}
		hiddenNodeDeltas[numUnfolds - 1] = hiddenNodeDeltasFromOutputNodeDeltas(outputNodeDeltas, hiddenNetValues[numUnfolds-1]);
		contextNodeDeltas[numUnfolds - 1] = contextNodeDeltasFromHiddenNodeDeltas(hiddenNodeDeltas[numUnfolds - 1]);
		for(int i = numUnfolds-2; i >= 0; i -= 1) {
			hiddenNodeDeltas[i] = hiddenNodeDeltasFromContextNodeDeltas(contextNodeDeltas[i+1], hiddenNetValues[i]);
			contextNodeDeltas[i] = contextNodeDeltasFromHiddenNodeDeltas(hiddenNodeDeltas[i]);
		}
		
		//change the supposed weights, if they may so be called
		
		
		return sum;
	}
}
