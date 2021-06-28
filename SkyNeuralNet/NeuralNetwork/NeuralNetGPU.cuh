#pragma once

#include <vector>
#include "NeuralNetGPUFunctions.cuh"

class NeuralNetGPU
{
private:
	void safeMalloc(const cudaError_t& error);
	void safeCopy(const cudaError_t& error);

public:
	void forwardProp(std::vector<double>& inputValues);
	void release();
};