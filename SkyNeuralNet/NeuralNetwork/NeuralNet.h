#pragma once

#include "Layer.h"
#include "NeuralNetGPU.cuh"

class NeuralNet
{
private:
	std::vector<Layer*> layers;

	NeuralNetGPU gpuNeuralNet;

	bool useGPU;

	void(NeuralNet::*forwardPropExecutionFunction)();

	void executeCudaForwardProp();
	void executeCPUForwardProp();

	void calcOutputLayerGradients(const std::vector<double>& expectedValues);
	void calcHiddenLayerGradients(const std::vector<double>& expectedValues);
	void updateWeights();

	const void getNeuronInfo(
		unsigned int& numNeurons,
		unsigned int& numWeights,
		unsigned int& maxNumNeuronsInLayer
	) const;

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