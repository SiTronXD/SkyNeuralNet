#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void cudaForwardProp(
	double* neuronOutputs,
	double* neuronWeights,
	int* neuronsPerLayer,
	int numLayers
);