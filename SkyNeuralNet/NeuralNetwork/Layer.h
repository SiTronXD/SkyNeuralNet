#pragma once

#include "Neuron.h"

class Layer
{
private:
	std::vector<Neuron*> neurons;

	Layer* previousLayer;

	double(*activationFunction)(double);

public:
	Layer(unsigned int numNeurons, unsigned int numOutputWeights, 
		Layer* previousLayer, bool isOutputLayer);
	~Layer();

	void setAllOutputs(std::vector<double> outputValues);
	void setAllOutputs(double* outputValues);
	void calcOutputs();
	void calcHiddenNeuronGradients(Layer* nextLayer);
	void calcOutputNeuronGradients(const std::vector<double>& expectedValues);
	void updateAllWeights(Layer* nextLayer);

	std::vector<Neuron*>& getNeurons();
};