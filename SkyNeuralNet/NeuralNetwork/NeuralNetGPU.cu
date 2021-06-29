#include "NeuralNetGPU.cuh"
#include <iostream>
#include "ActivationFunction.h"

void NeuralNetGPU::safeMalloc(const cudaError_t& error)
{
	if (error != cudaSuccess)
	{
		std::cout << "Cuda malloc failed..." << std::endl;
	}
}

void NeuralNetGPU::safeCopy(const cudaError_t& error)
{
	if (error != cudaSuccess)
	{
		std::cout << "Cuda copy failed..." << std::endl;
	}
}

void NeuralNetGPU::forwardProp(
	std::vector<Layer*>& layers, 
	const unsigned int numNeurons,
	const unsigned int numWeights,
	const unsigned int maxNumNeuronsInLayer
)
{
	// All outputs
	double* host_neuronOutputs = new double[numNeurons];
	double* devi_neuronOutputs = nullptr;

	// All weights
	double* host_neuronWeights = new double[numWeights];
	double* devi_neuronWeights = nullptr;

	// Number of neurons per layer
	int* host_neuronsPerLayer = new int[layers.size()];
	int* devi_neuronsPerLayer = nullptr;

	// Number of layers
	int host_numLayers = layers.size();


	// Neuron outputs
	unsigned int currentNeuron = 0;
	for (int i = 0; i < layers.size(); ++i)
	{
		std::vector<Neuron*>& layerNeurons = layers[i]->getNeurons();

		// Insert output neuron output values
		for (int j = 0; j < layerNeurons.size(); ++j)
		{
			host_neuronOutputs[currentNeuron++] = layerNeurons[j]->getOutputValue();
		}
	}
	this->safeMalloc(
		cudaMalloc(&devi_neuronOutputs, sizeof(double) * numNeurons)
	);
	this->safeCopy(
		cudaMemcpy(
			devi_neuronOutputs,
			host_neuronOutputs,
			sizeof(double) * numNeurons,
			cudaMemcpyHostToDevice
		)
	);

	// Neuron weights
	unsigned int currentWeight = 0;
	for (int i = 0; i < layers.size(); ++i)
	{
		std::vector<Neuron*>& currentNeurons = layers[i]->getNeurons();

		for (int j = 0; j < currentNeurons.size(); ++j)
		{
			std::vector<double>& currentWeights = currentNeurons[j]->getWeights();

			for (int k = 0; k < currentWeights.size(); ++k)
			{
				host_neuronWeights[currentWeight++] = currentWeights[k];
			}
		}
	}
	this->safeMalloc(
		cudaMalloc(&devi_neuronWeights, sizeof(double) * numWeights)
	);
	this->safeCopy(
		cudaMemcpy(
			devi_neuronWeights,
			host_neuronWeights,
			sizeof(double) * numWeights,
			cudaMemcpyHostToDevice
		)
	);

	// Number of neurons per layer
	for (int i = 0; i < layers.size(); ++i)
	{
		host_neuronsPerLayer[i] = layers[i]->getNeurons().size();
	}
	this->safeMalloc(
		cudaMalloc(&devi_neuronsPerLayer, sizeof(int) * layers.size())
	);
	this->safeCopy(
		cudaMemcpy(
			devi_neuronsPerLayer,
			host_neuronsPerLayer,
			sizeof(int) * layers.size(),
			cudaMemcpyHostToDevice
		)
	);

	// Execute on GPU
	cudaForwardProp<<<1, maxNumNeuronsInLayer >>>(
		devi_neuronOutputs, 
		devi_neuronWeights,
		devi_neuronsPerLayer,
		host_numLayers
	);
	cudaDeviceSynchronize();

	// Extract results
	this->safeCopy(
		cudaMemcpy(
			host_neuronOutputs,
			devi_neuronOutputs,
			sizeof(double) * numNeurons,
			cudaMemcpyDeviceToHost
		)
	);

	// Apply results
	unsigned int currentNeuronStride = layers[0]->getNeurons().size();
	for (int i = 1; i < layers.size(); ++i)
	{
		std::vector<double> results;
		std::vector<Neuron*>& currentNeurons = layers[i]->getNeurons();

		// Add result to vector
		for (int j = 0; j < currentNeurons.size() - 1; ++j)
		{
			double currentResult = host_neuronOutputs[currentNeuronStride + j];

			if (i == layers.size() - 1)
				currentResult = ActivationFunction::sigmoid(currentResult);

			results.push_back(currentResult);
		}

		layers[i]->setAllOutputs(results);
		currentNeuronStride += layers[i]->getNeurons().size();
	}

	delete[] host_neuronOutputs;
	delete[] host_neuronWeights;
	delete[] host_neuronsPerLayer;

	cudaFree(devi_neuronOutputs);
	cudaFree(devi_neuronWeights);
	cudaFree(devi_neuronsPerLayer);
}

void NeuralNetGPU::release()
{
	cudaDeviceReset();
}