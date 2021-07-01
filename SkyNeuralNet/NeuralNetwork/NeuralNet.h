#pragma once

#include <string>
#include "Layer.h"
#include "NeuralNetGPU.cuh"

class NeuralNet
{
private:
	std::vector<Layer*> layers;

	NeuralNetGPU gpuNeuralNet;

	void(NeuralNet::*forwardPropExecutionFunction)();
	void(NeuralNet::*backPropExecutionFunction)(const std::vector<double>& expectedValues);

	// Forward prop CUDA/CPU
	void executeCudaForwardProp();
	void executeCPUForwardProp();

	// Back prop CUDA/CPU
	void executeCudaBackProp(const std::vector<double>& expectedValues);
	void executeCPUBackProp(const std::vector<double>& expectedValues);

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
	void setUseGPUForwardProp(bool useGPU);
	void setUseGPUBackProp(bool useGPU);
	void outputNeuralNetToFile(const std::string outputPath);

	double calcError(const std::vector<double>& expectedValues) const;

	inline std::vector<Layer*>& getLayers() { return this->layers; }
};