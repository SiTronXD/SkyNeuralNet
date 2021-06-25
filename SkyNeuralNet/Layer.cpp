#include "Layer.h"
#include "ActivationFunction.h"

Layer::Layer(unsigned int numNeurons, unsigned int numOutputWeights,
	Layer* _previousLayer)
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
		this->neurons.push_back(new Neuron(initialOutputValue, numOutputWeights, newNeuronIndex));
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

		result = ActivationFunction::function(result);

		this->neurons[i]->setOutputValue(result);
	}
}

std::vector<Neuron*>& Layer::getNeurons()
{
	return this->neurons;
}
