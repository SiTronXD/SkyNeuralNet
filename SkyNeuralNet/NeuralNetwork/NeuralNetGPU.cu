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

NeuralNetGPU::NeuralNetGPU()
	: numNeurons(0), host_neuronOutputs(nullptr), devi_neuronOutputs(nullptr)
{
}

NeuralNetGPU::~NeuralNetGPU() { }

void NeuralNetGPU::setupTrainingSession(
	const unsigned int numLayers,
	const unsigned int numNeurons,
	const unsigned int numWeights,
	const unsigned int maxNumNeuronsInLayer
)
{
	this->numLayers = numLayers;
	this->numNeurons = numNeurons;
	this->numWeights = numWeights;
	this->maxNumNeuronsInLayer = maxNumNeuronsInLayer;

	// Variables for CPU <-> GPU communication
	this->host_neuronOutputs = new double[this->numNeurons];
	this->devi_neuronOutputs = nullptr;

	// All weights
	this->host_neuronWeights = new double[this->numWeights];
	this->devi_neuronWeights = nullptr;

	// Number of neurons per layer
	this->host_neuronsPerLayer = new int[this->numLayers];
	this->devi_neuronsPerLayer = nullptr;


	// Allocate variables on GPU
	this->safeMalloc(
		cudaMalloc(&devi_neuronOutputs, sizeof(double) * this->numNeurons)
	);
	this->safeMalloc(
		cudaMalloc(&devi_neuronWeights, sizeof(double) * this->numWeights)
	);
	this->safeMalloc(
		cudaMalloc(&devi_neuronsPerLayer, sizeof(int) * this->numLayers)
	);
}

void NeuralNetGPU::forwardProp(std::vector<Layer*>& layers)
{
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
	this->safeCopy(
		cudaMemcpy(
			devi_neuronsPerLayer,
			host_neuronsPerLayer,
			sizeof(int) * layers.size(),
			cudaMemcpyHostToDevice
		)
	);

	// ----- Execute on GPU -----
	cudaForwardProp<<<1, maxNumNeuronsInLayer >>>(
		devi_neuronOutputs, 
		devi_neuronWeights,
		devi_neuronsPerLayer,
		(int) this->numLayers
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

	// Let the CPU calculate activation function for output layer
	std::vector<Neuron*>& lastLayerNeurons = layers.back()->getNeurons();
	for (int i = 0; i < lastLayerNeurons.size(); ++i)
	{
		int currentIndex = this->numNeurons - lastLayerNeurons.size() + i;

		host_neuronOutputs[currentIndex] = 
			ActivationFunction::activateOutput(host_neuronOutputs[currentIndex]);
	}


	// Apply results to network
	unsigned int currentNeuronStride = layers[0]->getNeurons().size();
	for (int i = 1; i < layers.size(); ++i)
	{
		std::vector<double> results;
		std::vector<Neuron*>& currentNeurons = layers[i]->getNeurons();

		// Add result to vector
		for (int j = 0; j < currentNeurons.size() - 1; ++j)
		{
			double currentResult = host_neuronOutputs[currentNeuronStride + j];

			results.push_back(currentResult);
		}

		layers[i]->setAllOutputs(results);
		currentNeuronStride += layers[i]->getNeurons().size();
	}
}

void NeuralNetGPU::releaseTrainingSession()
{
	delete[] host_neuronOutputs;
	delete[] host_neuronWeights;
	delete[] host_neuronsPerLayer;

	cudaFree(devi_neuronOutputs);
	cudaFree(devi_neuronWeights);
	cudaFree(devi_neuronsPerLayer);

	cudaDeviceReset();
}