#pragma once

#include "Neuron.h"

class Layer
{
private:
	std::vector<Neuron> neurons;

	Layer* previousLayer;

public:
	Layer(unsigned int numNeurons, unsigned int numOutputWeights, 
		bool shouldHaveBias, Layer* previousLayer);
	~Layer();
};