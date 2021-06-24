#pragma once

#include "Layer.h"

class NeuralNet
{
private:
	std::vector<Layer*> layers;

public:
	NeuralNet(const std::vector<unsigned int>& neuronPerLayer);
	~NeuralNet();

	void forwardProp(const std::vector<double>& inputValues);
	void getOutputs(std::vector<double>& outputValues);

	void setWeight(unsigned int layerIndex, unsigned int neuronIndex,
		const std::vector<double>& newWeights);
};