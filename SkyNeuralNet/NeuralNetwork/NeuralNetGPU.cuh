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

	// ----- Mainly used in forward prop -----
	double* host_neuronOutputs;
	double* devi_neuronOutputs;

	double* host_neuronWeights;
	double* devi_neuronWeights;

	int* host_neuronsPerLayer;
	int* devi_neuronsPerLayer;

	// ----- Mainly used in back prop -----
	double* host_neuronGradients;
	double* devi_neuronGradients;

	double* host_neuronDeltaWeights;
	double* devi_neuronDeltaWeights;

	// Lookup array to get the current neuron index, from global weight index
	int* host_thisNeuronIndices;
	int* devi_thisNeuronIndices;

	// Lookup array to get neuron index that a global weight is connected to,
	// from global weight index
	int* host_nextNeuronIndices;
	int* devi_nextNeuronIndices;

	void safeMalloc(const cudaError_t& error);
	void safeCopy(const cudaError_t& error);

public:
	NeuralNetGPU();
	~NeuralNetGPU();

	void allocateTrainingSession(
		std::vector<Layer*>& layers,
		const unsigned int numNeurons,
		const unsigned int numWeights,
		const unsigned int maxNumNeuronsInLayer
	);
	void initTrainingSession(
		std::vector<Layer*>& layers
	);
	void forwardProp(
		std::vector<Layer*>& layers, 
		const std::vector<double>& inputValues
	);
	void backProp(
		std::vector<Layer*>& layers,
		const std::vector<double>& expectedValues
	);
	void extractApplyResults(std::vector<Layer*>& layers);
};