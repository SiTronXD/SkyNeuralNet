#include "NeuralNetGPUFunctions.cuh"

__device__ double activationFunctionHidden(double x)
{
	// Relu
	return fmax(0.0, x);
}
__device__ double activationFunctionDerivativeHidden(double x)
{
	return x >= 0.0 ? 1.0 : 0.0;
}

__device__ double activationFunctionOutput(double x)
{
	// Sigmoid

	// As expected, exp() gives slightly different 
	// results when comparing CUDA exp() and std::exp()
	return 1.0 / (1.0 + exp(-x));
}
__device__ double activationFunctionDerivativeOutput(double x)
{
	double s = activationFunctionOutput(x);

	return s * (1.0 - s);
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
				double outVal = neuronOutputs[lastLayerIndexStride + n];
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

__global__ void cudaBackProp(
	double* neuronOutputs,
	double* neuronWeights,
	double* neuronGradients,
	int* neuronsPerLayer,
	int numLayers
)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// Stride points to last hidden layer
	int layerStride = 0;
	int nextLayerStride = 0;
	for (int i = 0; i < numLayers - 1 - 1; ++i)
		layerStride += neuronsPerLayer[i];
	nextLayerStride = layerStride + neuronsPerLayer[numLayers - 1 - 1];

	// Stride points to last hidden layer weights
	int weightStride = 0;
	for (int i = 0; i < numLayers - 1 - 1; ++i)
	{
		// += <number of neurons> * <number of weights for each neuron>
		weightStride += neuronsPerLayer[i] * (neuronsPerLayer[i + 1] - 1);
	}

	// Go through each hidden layer, back to front, 
	// starting from the last hidden layer
	for (int i = numLayers - 1 - 1; i > 0; --i)
	{
		// Make sure this thread can work
		if (id < neuronsPerLayer[i])
		{
			// Sum weight gradients
			double swg = 0.0;

			for (int j = 0; j < neuronsPerLayer[i + 1] - 1; ++j)
			{
				swg +=
					neuronWeights[weightStride + id + j] *
					neuronGradients[nextLayerStride + j];
			}

			neuronGradients[layerStride + id] =
				swg *
				activationFunctionDerivativeHidden(neuronOutputs[layerStride + id]);
		}

		nextLayerStride = layerStride;
		layerStride -= neuronsPerLayer[i - 1];
		weightStride -= neuronsPerLayer[i - 1] * (neuronsPerLayer[i] - 1);

		__syncthreads();
	}
}