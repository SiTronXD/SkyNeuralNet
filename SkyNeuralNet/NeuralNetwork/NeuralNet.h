#pragma once

#include "Layer.h"

class NeuralNet
{
private:
	std::vector<Layer*> layers;

	void calcOutputLayerGradients(const std::vector<double>& expectedValues);
	void calcHiddenLayerGradients(const std::vector<double>& expectedValues);
	void updateWeights();

public:
	NeuralNet(const std::vector<unsigned int>& neuronPerLayer);
	~NeuralNet();

	void forwardProp(const std::vector<double>& inputValues);
	void backProp(const std::vector<double>& expectedValues);
	void getOutputs(std::vector<double>& outputValues);

	void setWeight(unsigned int layerIndex, unsigned int neuronIndex,
		const std::vector<double>& newWeights);

	double getError(const std::vector<double>& expectedValues) const;
};