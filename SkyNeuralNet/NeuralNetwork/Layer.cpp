#include "Layer.h"
#include "ActivationFunction.h"

Layer::Layer(unsigned int numNeurons, unsigned int numOutputWeights,
	Layer* _previousLayer, bool isOutputLayer)
	: previousLayer(_previousLayer)
{
	// Add neurons + 1 bias neuron
	for (int i = 0; i < numNeurons + 1; ++i)
	{
		double initialOutputValue = 0.0;

		// Bias neuron
		if (i == numNeurons)
			initialOutputValue = 1.0;

		// Add neuron
		int newNeuronIndex = this->neurons.size();
		this->neurons.push_back(
			new Neuron(
				initialOutputValue, 
				numOutputWeights, 
				newNeuronIndex,
				isOutputLayer
			)
		);
	}

	// Choose activation function
	if (isOutputLayer)
		this->activationFunction = ActivationFunction::activateOutput;
	else
		this->activationFunction = ActivationFunction::activateHidden;
}

Layer::~Layer()
{
	for (int i = 0; i < this->neurons.size(); ++i)
		delete this->neurons[i];
	this->neurons.clear();
}

// Sets output values for the input layer
void Layer::setAllOutputs(std::vector<double> outputValues)
{
	// Make sure the sizes are correct
	if (outputValues.size() != neurons.size() - 1)
		throw "Number of output values was not equal to number of neurons...";

	// Loop through neurons and set values
	for (int i = 0; i < neurons.size() - 1; ++i)
		this->neurons[i]->setOutputValue(outputValues[i]);
}

// Sets output values for the input layer
void Layer::setAllOutputs(double* outputValues)
{
	// Loop through neurons and set values
	for (int i = 0; i < this->neurons.size() - 1; ++i)
		this->neurons[i]->setOutputValue(outputValues[i]);
}

// Calculate outputs when executing forward propagation
void Layer::calcOutputs()
{
	// Go through each neuron, except for the bias
	for (int i = 0; i < this->neurons.size() - 1; ++i)
	{
		double result = 0.0;

		// Go through each neuron in the previous layer
		for (int j = 0; j < this->previousLayer->getNeurons().size(); ++j)
		{
			Neuron* prevNeuron = this->previousLayer->getNeurons()[j];

			result += 
				prevNeuron->getOutputValue() * 
				prevNeuron->getOutputWeight(i);
		}

		//result = ActivationFunction::function(result);
		result = this->activationFunction(result);

		this->neurons[i]->setOutputValue(result);
	}
}

void Layer::calcHiddenNeuronGradients(Layer* nextLayer)
{
	// Go through all neurons and calculate gradients
	for (int i = 0; i < this->neurons.size(); ++i)
		this->neurons[i]->calcHiddenGradient(nextLayer->getNeurons());
}

void Layer::calcOutputNeuronGradients(const std::vector<double>& expectedValues)
{
	// Calculate gradients for all output neurons, except bias neuron
	for (int i = 0; i < this->neurons.size() - 1; ++i)
		this->neurons[i]->calcOutputGradient(expectedValues[i]);
}

void Layer::updateAllWeights(Layer* nextLayer)
{
	// Go through all neurons and update all weights
	for (int i = 0; i < this->neurons.size(); ++i)
		this->neurons[i]->updateWeights(nextLayer);
}

std::vector<Neuron*>& Layer::getNeurons() { return this->neurons; }
