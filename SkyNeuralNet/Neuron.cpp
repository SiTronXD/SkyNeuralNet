#include "Neuron.h"

Neuron::Neuron(double initialOutputValue, unsigned int numOutputWeights)
	: outputValue(initialOutputValue)
{
	// Create random output weights
	for (int i = 0; i < numOutputWeights; ++i)
		outputWeights.push_back((double) rand() / RAND_MAX);
}

Neuron::~Neuron()
{
}
