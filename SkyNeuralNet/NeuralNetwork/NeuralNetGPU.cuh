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

	// Used in forward prop
	double* host_neuronOutputs;
	double* devi_neuronOutputs;

	double* host_neuronWeights;
	double* devi_neuronWeights;

	int* host_neuronsPerLayer;
	int* devi_neuronsPerLayer;

	// Used in back prop
	double* host_neuronGradients;
	double* devi_neuronGradients;

	double* host_neuronDeltaWeights;
	double* devi_neuronDeltaWeights;

	void safeMalloc(const cudaError_t& error);
	void safeCopy(const cudaError_t& error);

public:
	NeuralNetGPU();
	~NeuralNetGPU();

	void setupTrainingSession(
		std::vector<Layer*>& layers,
		const unsigned int numNeurons,
		const unsigned int numWeights,
		const unsigned int maxNumNeuronsInLayer
	);
	void forwardProp(std::vector<Layer*>& layers);
	void backProp(
		std::vector<Layer*>& layers, 
		const std::vector<double>& expectedValues
	);
	void releaseTrainingSession();
};