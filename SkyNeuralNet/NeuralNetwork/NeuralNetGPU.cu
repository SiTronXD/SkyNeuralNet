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
	: numLayers(0), numNeurons(0), numWeights(0), maxNumNeuronsInLayer(0),
	host_neuronOutputs(nullptr), devi_neuronOutputs(nullptr),
	host_neuronWeights(nullptr), devi_neuronWeights(nullptr),
	host_neuronsPerLayer(nullptr), devi_neuronsPerLayer(nullptr),

	host_neuronGradients(nullptr),
	devi_neuronGradients(nullptr),
	host_neuronDeltaWeights(nullptr),
	devi_neuronDeltaWeights(nullptr),
	host_thisNeuronIndices(nullptr),
	devi_thisNeuronIndices(nullptr),
	host_nextNeuronIndices(nullptr),
	devi_nextNeuronIndices(nullptr)
{
}

NeuralNetGPU::~NeuralNetGPU() { }

void NeuralNetGPU::setupTrainingSession(
	std::vector<Layer*>& layers,
	const unsigned int numNeurons,
	const unsigned int numWeights,
	const unsigned int maxNumNeuronsInLayer
)
{
	this->numLayers = layers.size();
	this->numNeurons = numNeurons;
	this->numWeights = numWeights;
	this->maxNumNeuronsInLayer = maxNumNeuronsInLayer;


	// ----- Variables for CPU <-> GPU communication -----

	// Global output values
	this->host_neuronOutputs = new double[this->numNeurons]{ };
	this->devi_neuronOutputs = nullptr;

	// Global weights
	this->host_neuronWeights = new double[this->numWeights]{ };
	this->devi_neuronWeights = nullptr;

	// Insert initial weights
	unsigned int currentWeightIndex = 0;
	for (int i = 0; i < layers.size(); ++i)
	{
		// Loop through neurons
		std::vector<Neuron*>& layerNeurons = layers[i]->getNeurons();
		for (int j = 0; j < layerNeurons.size(); ++j)
		{
			// Loop through weights
			std::vector<double>& currentWeights = layerNeurons[j]->getWeights();
			for (int k = 0; k < currentWeights.size(); ++k)
			{
				// Insert weights
				this->host_neuronWeights[currentWeightIndex++] = currentWeights[k];
			}
		}
	}

	// Number of neurons per layer
	this->host_neuronsPerLayer = new int[this->numLayers]{ };
	this->devi_neuronsPerLayer = nullptr;

	// Insert number of neurons per layer
	for(int i = 0; i < this->numLayers; ++i)
		host_neuronsPerLayer[i] = layers[i]->getNeurons().size();


	// Global gradients
	this->host_neuronGradients = new double[this->numNeurons]{ };
	this->devi_neuronGradients = nullptr;

	// Global delta weights
	this->host_neuronDeltaWeights = new double[this->numWeights]{ };
	this->devi_neuronDeltaWeights = nullptr;

	// Lookup array to get the current neuron index, from global weight index
	this->host_thisNeuronIndices = new int[this->numWeights]{ };
	this->devi_thisNeuronIndices = nullptr;

	// Lookup array to get neuron index that the global weight is connected to,
	// from global weight index
	this->host_nextNeuronIndices = new int[this->numWeights]{ };
	this->devi_nextNeuronIndices = nullptr;

	// Set lookup indices
	currentWeightIndex = 0;
	unsigned int thisNeuronStride = 0;
	unsigned int nextNeuronStride = 0;
	for (int i = 0; i < layers.size(); ++i)
	{
		// Loop through neurons
		std::vector<Neuron*>& currentNeurons = layers[i]->getNeurons();
		nextNeuronStride += currentNeurons.size();
		for (int j = 0; j < currentNeurons.size(); ++j)
		{
			// Loop through weights
			std::vector<double>& currentWeights = currentNeurons[j]->getWeights();
			for (int k = 0; k < currentWeights.size(); ++k)
			{
				this->host_thisNeuronIndices[currentWeightIndex] = thisNeuronStride;
				this->host_nextNeuronIndices[currentWeightIndex] = nextNeuronStride + k;

				currentWeightIndex++;
			}

			thisNeuronStride++;
		}
	}

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


	this->safeMalloc(
		cudaMalloc(&devi_neuronGradients, sizeof(double) * this->numNeurons)
	);
	this->safeMalloc(
		cudaMalloc(&devi_neuronDeltaWeights, sizeof(double) * this->numWeights)
	);
	this->safeMalloc(
		cudaMalloc(&devi_thisNeuronIndices, sizeof(double) * this->numWeights)
	);
	this->safeMalloc(
		cudaMalloc(&devi_nextNeuronIndices, sizeof(double) * this->numWeights)
	);

	// Copy over number of neurons per layer, 
	// since this stays static
	this->safeCopy(
		cudaMemcpy(
			devi_neuronsPerLayer,
			host_neuronsPerLayer,
			sizeof(int) * layers.size(),
			cudaMemcpyHostToDevice
		)
	);

	// Copy over initial weights once, then keep them on the GPU
	this->safeCopy(
		cudaMemcpy(
			this->devi_neuronWeights,
			this->host_neuronWeights,
			sizeof(double) * this->numWeights,
			cudaMemcpyHostToDevice
		)
	);

	// Copy over neuron indices,
	// since these stays static
	this->safeCopy(
		cudaMemcpy(
			this->devi_thisNeuronIndices,
			this->host_thisNeuronIndices,
			sizeof(double) * this->numWeights,
			cudaMemcpyHostToDevice
		)
	); 
	this->safeCopy(
		cudaMemcpy(
			this->devi_nextNeuronIndices,
			this->host_nextNeuronIndices,
			sizeof(double) * this->numWeights,
			cudaMemcpyHostToDevice
		)
	);
}

void NeuralNetGPU::forwardProp(
	std::vector<Layer*>& layers, 
	const std::vector<double>& inputValues
)
{
	// Get neuron outputs from input layer
	memcpy(this->host_neuronOutputs, &inputValues[0], sizeof(double) * inputValues.size());

	this->safeCopy(
		cudaMemcpy(
			this->devi_neuronOutputs,
			this->host_neuronOutputs,
			sizeof(double) * this->numNeurons,
			cudaMemcpyHostToDevice
		)
	);

	// ----- Execute on GPU -----
	cudaForwardProp<<<1, this->maxNumNeuronsInLayer>>>(
		this->devi_neuronOutputs, 
		this->devi_neuronWeights,
		this->devi_neuronsPerLayer,
		(int) this->numLayers
	);
	cudaDeviceSynchronize();

	// Extract results
	this->safeCopy(
		cudaMemcpy(
			this->host_neuronOutputs,
			this->devi_neuronOutputs,
			sizeof(double) * this->numNeurons,
			cudaMemcpyDeviceToHost
		)
	);

	// Let the CPU calculate activation function for output layer.
	// This is to keep precision when using std::exp().
	std::vector<Neuron*>& lastLayerNeurons = layers.back()->getNeurons();
	int currentIndex = this->numNeurons - lastLayerNeurons.size();
	for (int i = 0; i < lastLayerNeurons.size(); ++i)
	{
		host_neuronOutputs[currentIndex++] = 
			ActivationFunction::activateOutput(host_neuronOutputs[currentIndex]);
	}

	// ----- Apply output results to network -----
	unsigned int currentNeuronStride = this->numNeurons - layers.back()->getNeurons().size();
	layers.back()->setAllOutputs(&host_neuronOutputs[currentNeuronStride]);
}

void NeuralNetGPU::backProp(
	std::vector<Layer*>& layers, 
	const std::vector<double>& expectedValues
)
{
	// Calculate gradients in output layer 
	// (on the CPU, to keep precision when calculating the derivative)
	layers.back()->calcOutputNeuronGradients(expectedValues);

	// Get gradients from output layer
	std::vector<Neuron*>& currentNeurons = layers.back()->getNeurons();
	unsigned int numLastNeurons = currentNeurons.size();
	unsigned int neuronIndex = this->numNeurons - numLastNeurons;
	for (int j = 0; j < currentNeurons.size(); ++j)
	{
		// Insert gradients
		this->host_neuronGradients[neuronIndex++] = currentNeurons[j]->getGradient();
	}

	this->safeCopy(
		cudaMemcpy(
			this->devi_neuronGradients,
			this->host_neuronGradients,
			sizeof(double) * this->numNeurons,
			cudaMemcpyHostToDevice
		)
	);

	// ----- Execute on GPU -----
	cudaCalcGradients<<<1, this->maxNumNeuronsInLayer>>>(
		this->devi_neuronOutputs,
		this->devi_neuronWeights, 
		this->devi_neuronGradients,
		this->devi_neuronsPerLayer,
		(int) this->numLayers
	);
	cudaDeviceSynchronize();
	cudaUpdateWeights<<<(this->numWeights / 1024) + 1, 1024>>>(
		this->devi_neuronOutputs,
		this->devi_neuronWeights,
		this->devi_neuronDeltaWeights,
		this->devi_neuronGradients,
		this->devi_thisNeuronIndices,
		this->devi_nextNeuronIndices,
		(int) this->numWeights,
		Neuron::getETA(),
		Neuron::getALPHA()
	);
	cudaDeviceSynchronize();
}

void NeuralNetGPU::extractApplyResults(std::vector<Layer*>& layers)
{
	// Extract results

	// Gradients
	this->safeCopy(
		cudaMemcpy(
			this->host_neuronGradients,
			this->devi_neuronGradients,
			sizeof(double) * this->numNeurons,
			cudaMemcpyDeviceToHost
		)
	);
	// Weights
	this->safeCopy(
		cudaMemcpy(
			this->host_neuronWeights,
			this->devi_neuronWeights,
			sizeof(double) * this->numWeights,
			cudaMemcpyDeviceToHost
		)
	);
	// Delta weights
	this->safeCopy(
		cudaMemcpy(
			this->host_neuronDeltaWeights,
			this->devi_neuronDeltaWeights,
			sizeof(double) * this->numWeights,
			cudaMemcpyDeviceToHost
		)
	);

	// Apply results to network
	// Gradients, weights, delta weights
	unsigned int neuronIndex = 0;
	unsigned int weightIndex = 0;
	for (int i = 0; i < layers.size(); ++i)
	{
		std::vector<Neuron*>& currentNeurons = layers[i]->getNeurons();

		for (int j = 0; j < currentNeurons.size(); ++j)
		{
			// Set gradient
			currentNeurons[j]->setGradient(
				this->host_neuronGradients[neuronIndex++]
			);

			// Set weights and delta weights
			for (int k = 0; k < currentNeurons[j]->getWeights().size(); ++k)
			{
				currentNeurons[j]->setWeight(
					k,
					this->host_neuronWeights[weightIndex],
					this->host_neuronDeltaWeights[weightIndex]
				);

				weightIndex++;
			}
		}
	}
}

void NeuralNetGPU::releaseTrainingSession()
{
	delete[] this->host_neuronOutputs;
	delete[] this->host_neuronWeights;
	delete[] this->host_neuronsPerLayer;

	delete[] this->host_neuronGradients;
	delete[] this->host_neuronDeltaWeights;
	delete[] this->host_thisNeuronIndices;
	delete[] this->host_nextNeuronIndices;

	cudaFree(this->devi_neuronOutputs);
	cudaFree(this->devi_neuronWeights);
	cudaFree(this->devi_neuronsPerLayer);

	cudaFree(this->devi_neuronGradients);
	cudaFree(this->devi_neuronDeltaWeights);
	cudaFree(this->devi_thisNeuronIndices);
	cudaFree(this->devi_nextNeuronIndices);

	cudaDeviceReset();
}