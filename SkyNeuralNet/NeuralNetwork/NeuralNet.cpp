#include "NeuralNet.h"
#include <cmath>

void NeuralNet::calcOutputLayerGradients(const std::vector<double>& expectedValues)
{
	// Output layer
	this->layers.back()->calcOutputNeuronGradients(expectedValues);
}

void NeuralNet::calcHiddenLayerGradients(const std::vector<double>& expectedValues)
{
	// Go through all hidden layers, back to front
	for (int i = this->layers.size() - 2; i > 0; --i)
	{
		Layer* currentLayer = this->layers[i];
		Layer* nextLayer = this->layers[i + 1];

		currentLayer->calcHiddenNeuronGradients(nextLayer);
	}
}

void NeuralNet::updateWeights()
{
	// Go through all layers, except output layer,
	// and update all weights
	for (int i = this->layers.size() - 2; i >= 0; --i)
	{
		Layer* currentLayer = this->layers[i];
		Layer* nextLayer = this->layers[i + 1];

		currentLayer->updateAllWeights(nextLayer);
	}
}

NeuralNet::NeuralNet(const std::vector<unsigned int>& neuronPerLayer)
	: useGPU(true)
{
	for (int i = 0; i < neuronPerLayer.size(); ++i)
	{
		// Get number of output weights, if it exists
		int numOutputWeights = 0;
		if (i < neuronPerLayer.size() - 1)
			numOutputWeights = neuronPerLayer[i + 1];

		// Get previous layer, if it exists
		Layer* previousLayer = nullptr;
		if (i > 0)
			previousLayer = this->layers[i - 1];

		// Add layer
		this->layers.push_back(
			new Layer(
				neuronPerLayer[i], 
				numOutputWeights, 
				previousLayer,
				i == neuronPerLayer.size() - 1
			)
		);
	}
}

NeuralNet::~NeuralNet()
{
	for (int i = 0; i < this->layers.size(); ++i)
		delete this->layers[i];
	this->layers.clear();

	//this->gpuNeuralNet.release();
}

// Calculate output values in each layer
void NeuralNet::forwardProp(std::vector<double>& inputValues)
{
	// CUDA
	if (this->useGPU)	
	{
		this->gpuNeuralNet.forwardProp(inputValues);
	}
	// CPU
	else				
	{
		// Manually set output values in the input layer
		this->layers[0]->setAllOutputs(inputValues);

		for (int i = 1; i < this->layers.size(); ++i)
		{
			this->layers[i]->calcOutputs();
		}
	}
}

void NeuralNet::backProp(const std::vector<double>& expectedValues)
{
	// Calculate gradients in output layer
	this->calcOutputLayerGradients(expectedValues);

	// Calculate gradients in hidden layers
	this->calcHiddenLayerGradients(expectedValues);

	// Finally update weights
	this->updateWeights();
}

void NeuralNet::getOutputs(std::vector<double>& outputValues)
{
	outputValues.clear();

	// Get all neurons
	std::vector<Neuron*>& outputNeurons = this->layers.back()->getNeurons();

	// Save output values in given vector
	for (int i = 0; i < outputNeurons.size(); ++i)
		outputValues.push_back(outputNeurons[i]->getOutputValue());
}

// Set individual weights, from a predefined neural network
void NeuralNet::setWeight(unsigned int layerIndex, unsigned int neuronIndex,
	const std::vector<double>& newWeights)
{
	Neuron* currentNeuron = this->layers[layerIndex]->getNeurons()[neuronIndex];

	for (int i = 0; i < currentNeuron->getWeights().size(); ++i)
		currentNeuron->getWeights()[i] = newWeights[i];
}

void NeuralNet::setUseGPU(bool useGPU)
{
	this->useGPU = useGPU;
}

double NeuralNet::getError(const std::vector<double>& expectedValues) const
{
	// Calculate error (using "Root Mean Square Error")
	double error = 0.0;
	std::vector<Neuron*>& outputNeurons = this->layers.back()->getNeurons();

	// Accumulate errors
	for (int i = 0; i < outputNeurons.size() - 1; ++i)
	{
		double deltaError = expectedValues[i] - outputNeurons[i]->getOutputValue();

		error += deltaError * deltaError;
	}
	error = std::sqrt(error / (outputNeurons.size() - 1));

	return error;
}