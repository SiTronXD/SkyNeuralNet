#include "NeuralNetGPU.cuh"
#include <iostream>

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

void NeuralNetGPU::forwardProp(std::vector<double>& inputValues)
{
	double* host_inputValues = new double[inputValues.size()];
	double* devi_inputValues = nullptr;

	this->safeMalloc(
		cudaMalloc(&devi_inputValues, sizeof(double) * inputValues.size())
	);
	this->safeCopy(
		cudaMemcpy(
			devi_inputValues,
			&inputValues[0],
			sizeof(double) * inputValues.size(),
			cudaMemcpyHostToDevice
		)
	);

	cudaForwardProp<<<1, 1>>>(devi_inputValues);

	this->safeCopy(
		cudaMemcpy(
			host_inputValues,
			devi_inputValues,
			sizeof(double) * inputValues.size(),
			cudaMemcpyDeviceToHost
		)
	);

	std::cout << "GPU ANSWER " << host_inputValues[0] << std::endl;

	delete[] host_inputValues;

	cudaFree(devi_inputValues);
}

void NeuralNetGPU::release()
{
	cudaDeviceReset();
}