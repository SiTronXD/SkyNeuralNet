#include "Layer.h"

Layer::Layer(unsigned int numNeurons, unsigned int numOutputWeights,
	bool shouldHaveBias, Layer* previousLayer)
	: previousLayer(previousLayer)
{
	// Add neurons + 1 bias neuron
	for (int i = 0; i < numNeurons + shouldHaveBias; ++i)
	{
		double initialOutputValue = (double) rand() / RAND_MAX;

		// Bias neuron
		if (i == numNeurons)
			initialOutputValue = 1.0;

		this->neurons.push_back(Neuron(initialOutputValue, numOutputWeights));
	}
}

Layer::~Layer()
{
}
