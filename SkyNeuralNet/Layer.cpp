#include "Layer.h"
#include <cmath>

Layer::Layer(unsigned int numNeurons, unsigned int numOutputWeights,
	Layer* _previousLayer)
	: previousLayer(_previousLayer)
{
	// Add neurons + 1 bias neuron
	for (int i = 0; i < numNeurons + 1; ++i)
	{
		double initialOutputValue = (double) rand() / RAND_MAX;

		// Bias neuron
		if (i == numNeurons)
			initialOutputValue = 1.0;

		// Add neuron
		this->neurons.push_back(new Neuron(initialOutputValue, numOutputWeights));
	}
}

Layer::~Layer()
{
	for (int i = 0; i < this->neurons.size(); ++i)
		delete this->neurons[i];
	this->neurons.clear();
}

void Layer::setAllOutputs(std::vector<double> outputValues)
{
	// Make sure the sizes are correct
	if (outputValues.size() != neurons.size() - 1)
		throw "Number of output values was not equal to number of neurons...";

	// Set all values
	for (int i = 0; i < outputValues.size(); ++i)
	{
		this->neurons[i]->setOutputValue(outputValues[i]);
	}
}

void Layer::calcOutputs()
{
	// Calcultae outputs for each neuron,
	// except for the bias
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

		result = this->activationFunction(result);

		this->neurons[i]->setOutputValue(result);
	}
}

double Layer::activationFunction(double x) const
{
	return std::tanh(x);
}

double Layer::activationFunctionDerivative(double x) const
{
	return 1.0 - (std::tanh(x) * std::tanh(x));
}

std::vector<Neuron*>& Layer::getNeurons()
{
	return this->neurons;
}
