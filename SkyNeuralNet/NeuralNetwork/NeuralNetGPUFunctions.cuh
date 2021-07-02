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

__global__ void cudaCalcGradients(
	double* neuronOutputs,
	double* neuronWeights,
	double* neuronGradients,
	int* neuronsPerLayer,
	int numLayers
);

__global__ void cudaUpdateWeights(
	double* neuronOutputs,
	double* neuronWeights,
	double* neuronDeltaWeights,
	double* neuronGradients,
	int* thisNeuronIndex,
	int* nextNeuronIndex,
	int numWeights,
	float eta,
	float alpha
);