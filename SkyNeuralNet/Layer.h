#pragma once

#include "Neuron.h"

class Layer
{
private:
	std::vector<Neuron*> neurons;

	Layer* previousLayer;

public:
	Layer(unsigned int numNeurons, unsigned int numOutputWeights, 
		Layer* previousLayer);
	~Layer();

	void setAllOutputs(std::vector<double> outputValues);
	void calcOutputs();

	std::vector<Neuron*>& getNeurons();
};