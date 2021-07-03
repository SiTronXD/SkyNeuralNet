#pragma once

#include <string>
#include "Layer.h"
#include "NeuralNetGPU.cuh"

class NeuralNet
{
private:
	std::vector<unsigned int> neuronsPerLayer;
	std::vector<Layer*> layers;

	NeuralNetGPU gpuNeuralNet;

	bool useGPU;

	void(NeuralNet::*forwardPropExecutionFunction)(std::vector<double>& inputValues);
	void(NeuralNet::*backPropExecutionFunction)(const std::vector<double>& expectedValues);

	// Forward prop CUDA/CPU
	void executeCudaForwardProp(std::vector<double>& inputValues);
	void executeCPUForwardProp(std::vector<double>& inputValues);

	// Back prop CUDA/CPU
	void executeCudaBackProp(const std::vector<double>& expectedValues);
	void executeCPUBackProp(const std::vector<double>& expectedValues);

	void calcOutputLayerGradients(const std::vector<double>& expectedValues);
	void calcHiddenLayerGradients();
	void updateWeights();

	void deallocateLayers();

	const void getNeuronInfo(
		unsigned int& numNeurons,
		unsigned int& numWeights,
		unsigned int& maxNumNeuronsInLayer
	) const;

	void setUseGPU(bool _useGPU);
	void setUseGPUForwardProp(bool _useGPU);
	void setUseGPUBackProp(bool _useGPU);

public:
	NeuralNet(std::vector<unsigned int> neuronPerLayer, bool _useGPU = true);
	~NeuralNet();

	void forwardProp(std::vector<double>& inputValues);
	void backProp(const std::vector<double>& expectedValues);
	void getOutputs(std::vector<double>& outputValues);

	void setWeight(unsigned int layerIndex, unsigned int neuronIndex,
		const std::vector<double>& newWeights);

	void outputNeuralNetToFile(const std::string outputPath);

	void resetNetStructure();
	void endTrainingSession();

	double calcError(const std::vector<double>& expectedValues) const;

	inline std::vector<Layer*>& getLayers() { return this->layers; }
};