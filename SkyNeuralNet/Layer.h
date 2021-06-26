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
	void calcHiddenNeuronGradients(Layer* nextLayer);
	void calcOutputNeuronGradients(const std::vector<double>& expectedValues);
	void updateAllWeights(Layer* nextLayer);

	std::vector<Neuron*>& getNeurons();
};