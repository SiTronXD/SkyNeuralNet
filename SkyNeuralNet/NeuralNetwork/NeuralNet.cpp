#include "NeuralNet.h"
#include <iostream>
#include <cmath>
#include <fstream>

// Execute forward propagation with CUDA
void NeuralNet::executeCudaForwardProp(std::vector<double>& inputValues)
{
	// Add input value for bias neuron
	inputValues.push_back(1.0);

	this->gpuNeuralNet.forwardProp(this->layers, inputValues);
}

// Execute forward propagation on the CPU
void NeuralNet::executeCPUForwardProp(std::vector<double>& inputValues)
{
	// Manually set output values in the input layer
	this->layers[0]->setAllOutputs(inputValues);

	// Calculate each output value
	for (int i = 1; i < this->layers.size(); ++i)
		this->layers[i]->calcOutputs();
}

// Execute back propagation with CUDA
void NeuralNet::executeCudaBackProp(const std::vector<double>& expectedValues)
{
	// Calculate gradients in output layer 
	// (on the CPU, to keep precision when calculating the derivative)
	this->calcOutputLayerGradients(expectedValues);

	// Calculate gradients in hidden layers 
	// and update weights (in CUDA)
	this->gpuNeuralNet.backProp(this->layers);
}

// Execute back propagation on the CPU
void NeuralNet::executeCPUBackProp(const std::vector<double>& expectedValues)
{
	// Calculate gradients in output layer
	this->calcOutputLayerGradients(expectedValues);

	// Calculate gradients in hidden layers
	this->calcHiddenLayerGradients();

	// Finally update weights
	this->updateWeights();
}

void NeuralNet::calcOutputLayerGradients(const std::vector<double>& expectedValues)
{
	// Output layer
	this->layers.back()->calcOutputNeuronGradients(expectedValues);
}

void NeuralNet::calcHiddenLayerGradients()
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
	for (int i = 0; i < this->layers.size() - 1; ++i)
	{
		Layer* currentLayer = this->layers[i];
		Layer* nextLayer = this->layers[i + 1];

		currentLayer->updateAllWeights(nextLayer);
	}
}

const void NeuralNet::getNeuronInfo(
	unsigned int& numNeurons, 
	unsigned int& numWeights, 
	unsigned int& maxNumNeuronsInLayer
) const
{
	numNeurons = 0;
	numWeights = 0;
	maxNumNeuronsInLayer = 0;

	for (int i = 0; i < this->layers.size(); ++i)
	{
		std::vector<Neuron*>& currentNeurons = this->layers[i]->getNeurons();
		unsigned int currentNumNeurons = currentNeurons.size();

		// Number of neurons
		numNeurons += currentNumNeurons;

		// Max number of neurons in layer
		if (currentNumNeurons > maxNumNeuronsInLayer)
			maxNumNeuronsInLayer = currentNumNeurons;

		// Number of weights
		for (int j = 0; j < currentNeurons.size(); ++j)
			numWeights += currentNeurons[j]->getWeights().size();
	}
}

NeuralNet::NeuralNet(const std::vector<unsigned int>& neuronPerLayer)
{
	this->setUseGPU(true);

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

	// Get info about network
	unsigned int numNeurons, numWeights, maxNumNeuronsInLayer;
	this->getNeuronInfo(numNeurons, numWeights, maxNumNeuronsInLayer);

	// Setup training session for the GPU
	this->gpuNeuralNet.setupTrainingSession(
		this->layers,
		numNeurons,
		numWeights,
		maxNumNeuronsInLayer
	);
}

NeuralNet::~NeuralNet()
{
	for (int i = 0; i < this->layers.size(); ++i)
		delete this->layers[i];
	this->layers.clear();

	this->gpuNeuralNet.releaseTrainingSession();
}

// Calculate output values in each layer
void NeuralNet::forwardProp(std::vector<double>& inputValues)
{
	// Execute
	(this->*forwardPropExecutionFunction)(inputValues);
}

void NeuralNet::backProp(const std::vector<double>& expectedValues)
{
	// Execute
	(this->*backPropExecutionFunction)(expectedValues);
}

void NeuralNet::getOutputs(std::vector<double>& outputValues)
{
	outputValues.clear();

	// Get all neurons
	std::vector<Neuron*>& outputNeurons = this->layers.back()->getNeurons();

	// Save output values in given vector
	for (int i = 0; i < outputNeurons.size() - 1; ++i)
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

	this->setUseGPUForwardProp(useGPU);
	this->setUseGPUBackProp(useGPU);
}

void NeuralNet::setUseGPUForwardProp(bool useGPU)
{
	// CUDA
	if (useGPU)
	{
		this->forwardPropExecutionFunction = &NeuralNet::executeCudaForwardProp;
	}
	// CPU
	else
	{
		this->forwardPropExecutionFunction = &NeuralNet::executeCPUForwardProp;
	}
}

void NeuralNet::setUseGPUBackProp(bool useGPU)
{
	// CUDA
	if (useGPU)
	{
		this->backPropExecutionFunction = &NeuralNet::executeCudaBackProp;
	}
	// CPU
	else
	{
		this->backPropExecutionFunction = &NeuralNet::executeCPUBackProp;
	}
}

void NeuralNet::outputNeuralNetToFile(const std::string outputPath)
{
	std::ofstream outputFile;

	outputFile.open(outputPath);


	// Number of layers
	outputFile << "NumLayers: " << this->layers.size() << std::endl;

	// Number of neurons per layer (ignore bias neurons)
	outputFile << "NumNeuronsPerLayer: ";
	for (int i = 0; i < this->layers.size(); ++i)
	{
		outputFile << this->layers[i]->getNeurons().size() - 1 << " ";
	}
	outputFile << std::endl << std::endl;
	outputFile << "# <layerIndex> <neuronIndex> <weightIndex> <value>" << std::endl;

	// Weights
	for (int i = 0; i < this->layers.size(); ++i)
	{
		std::vector<Neuron*> neurons = this->layers[i]->getNeurons();

		for (int j = 0; j < neurons.size(); ++j)
		{
			std::vector<double> weights = neurons[j]->getWeights();

			for (int k = 0; k < weights.size(); ++k)
			{
				outputFile << i << " " << j << " " << k << " " << 
					weights[k];

				// New line, if not last
				if (!(i == this->layers.size() - 1 - 1 && 
					j == neurons.size() - 1 && 
					k == weights.size() - 1))
					outputFile << std::endl;
			}
		}
	}

	outputFile.close();
}

void NeuralNet::endTrainingSession()
{
	// Extract and apply results from GPU
	if (this->useGPU)
	{
		this->gpuNeuralNet.extractApplyResults(this->layers);
	}
}

double NeuralNet::calcError(const std::vector<double>& expectedValues) const
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
