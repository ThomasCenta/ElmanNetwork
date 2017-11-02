import java.util.ArrayList;
import java.util.Random;

public class ElmanNetwork {
	
	private double contextInitialization() {
		return 0.0;
	}
	
	private static double edgeWeightInitialization() {
		double toReturn = 5*rand.nextDouble()+1;
		if(rand.nextBoolean()) {
			toReturn*= -1;
		}
		return toReturn;
	}
	
	
	private static void setEdgesHiddenNodeToOutputLayer(ArrayList<Edge> edgeList, HiddenNode hiddenNode, ArrayList<OutputNode> outputLayer) {
		for(OutputNode outputNode: outputLayer) {
			Edge edge = new Edge(hiddenNode, outputNode, edgeWeightInitialization());
			hiddenNode.getTrailingEdges().add(edge);
			outputNode.getLeadingEdges().add(edge);
			edgeList.add(edge);
		}
	}

	
	private static void setEdgesInputNodeToOutputLayer(ArrayList<Edge> edgeList, InputNode inputNode, ArrayList<OutputNode> outputLayer) {
		for(OutputNode outputNode: outputLayer) {
			Edge edge = new Edge(inputNode, outputNode, edgeWeightInitialization());
			outputNode.getLeadingEdges().add(edge);
			edgeList.add(edge);
		}
	}
	
	private static void setEdgesInputNodeToHiddenLayer(ArrayList<Edge> edgeList, InputNode inputNode, ArrayList<HiddenNode> hiddenLayer) {
		for(HiddenNode hiddenNode: hiddenLayer) {
			Edge edge = new Edge(inputNode, hiddenNode, edgeWeightInitialization());
			hiddenNode.getLeadingEdges().add(edge);
			edgeList.add(edge);
		}
	}
	
	private static void setEdgesContextNodeToHiddenLayer(ArrayList<Edge> edgeList, ContextNode contextNode, ArrayList<HiddenNode> hiddenLayer) {
		for(HiddenNode hiddenNode: hiddenLayer) {
			Edge edge = new Edge(contextNode, hiddenNode, edgeWeightInitialization());
			hiddenNode.getLeadingEdges().add(edge);
			contextNode.getTrailingEdges().add(edge);
			edgeList.add(edge);
		}
	}
	
	private static void setEdgesContextLayerToHiddenLayer(ArrayList<Edge> edgeList, ArrayList<ContextNode> contextLayer, ArrayList<HiddenNode> hiddenLayer) {
		for(ContextNode contextNode: contextLayer) {
			setEdgesContextNodeToHiddenLayer(edgeList, contextNode, hiddenLayer);
		}
	}
	
	// sets edges between the hidden and output layers, putting them into edgeList as they are made
	private static void setEdgesHiddenLayerToOutputLayer(ArrayList<Edge> edgeList, ArrayList<HiddenNode> hiddenLayer, ArrayList<OutputNode> outputLayer) {
		for(HiddenNode hiddenNode: hiddenLayer) {
			setEdgesHiddenNodeToOutputLayer(edgeList, hiddenNode, outputLayer);
		}
	}
	
	// sets edges between the hidden and output layers, putting them into edgeList as they are made
	private static void setEdgesInputLayerToHiddenLayer(ArrayList<Edge> edgeList, ArrayList<InputNode> inputLayer, ArrayList<HiddenNode> hiddenLayer) {
		for(InputNode inputNode: inputLayer) {
			setEdgesInputNodeToHiddenLayer(edgeList, inputNode, hiddenLayer);
		}
	}
	
	// puts an edge between a hidden node and its respective context node
	private static void setEdgesHiddenLayerToContextLayer(ArrayList<Edge> edgeList, ArrayList<HiddenNode> hiddenLayer, ArrayList<ContextNode> contextLayer) {
		assert hiddenLayer.size() == contextLayer.size();
		
		for(int i = 0; i < hiddenLayer.size(); i += 1) {
			Edge edge = new Edge(hiddenLayer.get(i), contextLayer.get(i), 1.0);
			hiddenLayer.get(i).getTrailingEdges().add(edge);
			contextLayer.get(i).getLeadingEdges().add(edge);
			edgeList.add(edge);
		}
	}
	
	
	private static Random rand = new Random();
	
	private ArrayList<InputNode> inputNodes;
	private ArrayList<ContextNode> contextNodes;
	private ArrayList<HiddenNode> hiddenNodes;
	private ArrayList<OutputNode> outputNodes;
	private InputNode hiddenBias;
	private InputNode outputBias;
	
	private ArrayList<Edge> hiddenOutEdges;
	private ArrayList<Edge> inputHiddenEdges;
	
	public ElmanNetwork(int numInputs, int numHiddenNodes, int numOutputs, int numTimesToUnfold, ActivationFunction hiddenActivationFunction, ActivationFunction outputActivationFunction, ErrorFunction errorFunction) {
		this.inputNodes = new ArrayList<InputNode>(numInputs);
		for(int i = 0; i < numInputs; i += 1) {
			this.inputNodes.add(new InputNode());
		}
		this.contextNodes = new ArrayList<ContextNode>(numHiddenNodes);
		for(int i = 0; i < numHiddenNodes; i += 1) {
			//just let them have linear activation functions
			this.contextNodes.add(new ContextNode());
		}
		this.hiddenNodes = new ArrayList<HiddenNode>(numHiddenNodes);
		for(int i = 0; i < numHiddenNodes; i += 1) {
			this.hiddenNodes.add(new HiddenNode(hiddenActivationFunction));
		}
		this.outputNodes = new ArrayList<OutputNode>(numOutputs);
		for(int i = 0; i < numOutputs; i += 1) {
			this.outputNodes.add(new OutputNode(errorFunction, outputActivationFunction));
		}
		this.hiddenBias.setOut(1);
		this.outputBias.setOut(1);
		
		//set all edges
		setEdgesInputLayerToHiddenLayer(this.inputHiddenEdges, this.inputNodes, this.hiddenNodes);
		setEdgesInputNodeToHiddenLayer(this.inputHiddenEdges, this.hiddenBias, this.hiddenNodes);
		setEdgesHiddenLayerToOutputLayer(this.hiddenOutEdges, this.hiddenNodes, this.outputNodes);
		setEdgesInputNodeToOutputLayer(this.hiddenOutEdges, this.outputBias, this.outputNodes);
	}
}
