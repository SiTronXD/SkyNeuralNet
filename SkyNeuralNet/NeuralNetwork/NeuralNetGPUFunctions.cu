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

__global__ void cudaForwardProp(
	double* neuronOutputs,
	double* neuronWeights,
	int* neuronsPerLayer,
	int numLayers
)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int layerIndexStride = 0;
	int lastLayerIndexStride = 0;
	int lastLayerWeightStride = 0;

	// Go through each layer
	for (int l = 1; l < numLayers; ++l)
	{
		layerIndexStride += neuronsPerLayer[l - 1];

		// Don't calculate output for bias neurons
		if (id < neuronsPerLayer[l] - 1)
		{
			neuronOutputs[layerIndexStride + id] = 0;

			// Go through each neuron from the last layer
			for (int n = 0; n < neuronsPerLayer[l - 1]; ++n)
			{
				neuronOutputs[layerIndexStride + id] += 
					// Output value
					neuronOutputs[lastLayerIndexStride + n] * 
					// Weight
					neuronWeights[
						lastLayerWeightStride +
						(neuronsPerLayer[l] - 1) * n + // Remove bias neuron
						id
					];
			}

			// Activation functions
			if (l < numLayers - 1)
			{
				neuronOutputs[layerIndexStride + id] =
					activationFunctionHidden(neuronOutputs[layerIndexStride + id]);
			}
			// Let the CPU handle sigmoid
			// (Takes 1 more second for 5000 training sets)
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

	/*if (neuronOutputs[0] != neuronOutputs[1])
		neuronOutputs[0] = 1;
	else
		neuronOutputs[0] = 0;*/
}