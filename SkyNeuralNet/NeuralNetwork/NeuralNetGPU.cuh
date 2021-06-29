#pragma once

#include <vector>
#include "NeuralNetGPUFunctions.cuh"
#include "Layer.h"

class NeuralNetGPU
{
private:
	void safeMalloc(const cudaError_t& error);
	void safeCopy(const cudaError_t& error);

public:
	void forwardProp(
		std::vector<Layer*>& layers, 
		const unsigned int numNeurons,
		const unsigned int numWeights,
		const unsigned int maxNumNeuronsInLayer
	);
	void release();
};