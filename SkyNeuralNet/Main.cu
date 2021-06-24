#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__global__ void AddIntsCUDA(int* a, int* b)
{
	a[0] += b[0];
}

int main()
{
	int a = 5, b = 9;
	int* d_a, * d_b;

	// Allocate memory on gpu
	if (cudaMalloc(&d_a, sizeof(int)) != cudaSuccess)
	{
		cout << "Error allocating memory!" << endl;
		return 1;
	}
	if (cudaMalloc(&d_b, sizeof(int)) != cudaSuccess)
	{
		cout << "Error allocating memory!" << endl;
		cudaFree(d_a);
		return 1;
	}

	// Copy values into allocated memory
	if (cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "Error copying memory!" << endl;
		cudaFree(d_a);
		cudaFree(d_b);
		return 1;
	}
	if (cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "Error copying memory!" << endl;
		cudaFree(d_a);
		cudaFree(d_b);
		return 1;
	}

	// Execute function on gpu
	AddIntsCUDA <<<1, 1 >>> (d_a, d_b);
	cudaDeviceSynchronize();

	// Copy calculated value from device to host
	if (cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cout << "Error copying memory!" << endl;
		cudaFree(d_a);
		cudaFree(d_b);
		return 1;
	}

	cout << "a: " << a << endl;

	cudaFree(d_a);
	cudaFree(d_b);

	cudaDeviceReset();

	getchar();

	return 0;
}