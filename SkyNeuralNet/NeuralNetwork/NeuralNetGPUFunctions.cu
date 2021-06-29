#include "NeuralNetGPUFunctions.cuh"

__device__ double activationFunctionHidden(double x)
{
	// Relu
	return fmax(0.0, x);
}

__device__ double activationFunctionOutput(double x)
{
	// Sigmoid

	// exp() gives slightly different results when comparing
	// CUDA exp() and std::exp()
	return 1.0 / (1.0 + exp(-x));
}

#define MAX_BLOCKING_NEURON_SIZE 1024

__global__ void cudaForwardProp(
	double* neuronOutputs,
	double* neuronWeights,
	int* neuronsPerLayer,
	int numLayers
)
{
	__shared__ double lastLayerOutputs[MAX_BLOCKING_NEURON_SIZE];

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int layerIndexStride = 0;
	int lastLayerIndexStride = 0;
	int lastLayerWeightStride = 0;

	// Go through each layer
	for (int l = 1; l < numLayers; ++l)
	{
		layerIndexStride += neuronsPerLayer[l - 1];

		// Load last layer output values
		// into shared memory
		if (id < neuronsPerLayer[l - 1])
		{
			lastLayerOutputs[id] = 
				neuronOutputs[lastLayerIndexStride + id];
		}
		__syncthreads();

		// Don't calculate output for bias neurons
		if (id < neuronsPerLayer[l] - 1)
		{
			neuronOutputs[layerIndexStride + id] = 0;

			// Go through each neuron from the last layer
			for (int n = 0; n < neuronsPerLayer[l - 1]; ++n)
			{
				double outVal = lastLayerOutputs[n];
				double weightVal = 
					neuronWeights[
						lastLayerWeightStride +
						(neuronsPerLayer[l] - 1) * n + // Ignore bias neuron
						id
					];

				neuronOutputs[layerIndexStride + id] += outVal * weightVal;
			}

			// Activation function for hidden layers
			if (l < numLayers - 1)
			{
				neuronOutputs[layerIndexStride + id] =
					activationFunctionHidden(neuronOutputs[layerIndexStride + id]);
			}
			// Activation function for output layer
			// (Let the CPU do it to keep precision,
			// only takes +1 second for 5000 training sets)
			/*else
			{
				neuronOutputs[layerIndexStride + id] =
					activationFunctionOutput(neuronOutputs[layerIndexStride + id]);
			}*/
		}
		// Bias neuron
		else if (id == neuronsPerLayer[l] - 1)
		{
			neuronOutputs[layerIndexStride + id] = 1.0;
		}

		lastLayerWeightStride += (neuronsPerLayer[l - 1]) * (neuronsPerLayer[l] - 1);
		lastLayerIndexStride = layerIndexStride;

		__syncthreads();
	}
}