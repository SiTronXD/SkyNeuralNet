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

	double activationFunction(double x) const;
	double activationFunctionDerivative(double x) const;

	std::vector<Neuron*>& getNeurons();
};