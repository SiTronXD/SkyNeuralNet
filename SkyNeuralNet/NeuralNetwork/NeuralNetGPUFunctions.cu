#include "NeuralNetGPUFunctions.cuh"

__global__ void cudaForwardProp(double* inputLayerValues)
{
	if (inputLayerValues[0] != inputLayerValues[1])
		inputLayerValues[0] = 1;
	else
		inputLayerValues[0] = 0;
}