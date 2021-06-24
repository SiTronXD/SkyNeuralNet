#include <iostream>

#include "NeuralNet.h"

int main()
{
	// Create neurons for each layer
	std::vector<unsigned int> neuronsPerLayer{ 2, 4, 1 };
	NeuralNet nn(neuronsPerLayer);

	getchar();

	return 0;
}