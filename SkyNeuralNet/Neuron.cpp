#include "Neuron.h"

Neuron::Neuron(double initialOutputValue, unsigned int numOutputWeights)
	: outputValue(initialOutputValue)
{
	// Create random output weights
	for (int i = 0; i < numOutputWeights; ++i)
		this->outputWeights.push_back((double) rand() / RAND_MAX);
}

Neuron::~Neuron()
{
}

void Neuron::setOutputValue(double outputValue)
{
	this->outputValue = outputValue;
}

double Neuron::getOutputValue() const
{
	return this->outputValue;
}

double Neuron::getOutputWeight(int index) const
{
	return this->outputWeights[index];
}

std::vector<double>& Neuron::getWeights()
{
	return this->outputWeights;
}
