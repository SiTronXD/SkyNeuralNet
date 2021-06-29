#pragma once

#include <vector>
#include "NeuralNetGPUFunctions.cuh"
#include "Layer.h"

class NeuralNetGPU
{
private:
	unsigned int numLayers;
	unsigned int numNeurons;
	unsigned int numWeights;
	unsigned int maxNumNeuronsInLayer;

	double* host_neuronOutputs;
	double* devi_neuronOutputs;

	double* host_neuronWeights;
	double* devi_neuronWeights;

	int* host_neuronsPerLayer;
	int* devi_neuronsPerLayer;

	void safeMalloc(const cudaError_t& error);
	void safeCopy(const cudaError_t& error);

public:
	NeuralNetGPU();
	~NeuralNetGPU();

	void setupTrainingSession(
		const unsigned int numLayers,
		const unsigned int numNeurons,
		const unsigned int numWeights,
		const unsigned int maxNumNeuronsInLayer
	);
	void forwardProp(std::vector<Layer*>& layers);
	void releaseTrainingSession();
};