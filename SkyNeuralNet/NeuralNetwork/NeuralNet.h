#pragma once

#include "Layer.h"
#include "NeuralNetGPU.cuh"

class NeuralNet
{
private:
	std::vector<Layer*> layers;

	NeuralNetGPU gpuNeuralNet;

	void calcOutputLayerGradients(const std::vector<double>& expectedValues);
	void calcHiddenLayerGradients(const std::vector<double>& expectedValues);
	void updateWeights();

	bool useGPU;

public:
	NeuralNet(const std::vector<unsigned int>& neuronPerLayer);
	~NeuralNet();

	void forwardProp(std::vector<double>& inputValues);
	void backProp(const std::vector<double>& expectedValues);
	void getOutputs(std::vector<double>& outputValues);

	void setWeight(unsigned int layerIndex, unsigned int neuronIndex,
		const std::vector<double>& newWeights);

	void setUseGPU(bool useGPU);

	double getError(const std::vector<double>& expectedValues) const;
};