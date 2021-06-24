#include <iostream>

#include "NeuralNet.h"

int main()
{
	// Catch memory leaks
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	// Create neurons for each layer
	std::vector<unsigned int> neuronsPerLayer{ 2, 4, 1 };
	NeuralNet nn(neuronsPerLayer);


	// DEBUGGING USING WEIGHTS FOR XOR
	std::vector<double> weights00{ 0.278197, 2.05765, 1.32282, 1.05197 };
	std::vector<double> weights01{ 0.494084, 2.06895, 1.25901, 1.11944 };
	std::vector<double> weights02{ 0.218981, -0.6279, -1.83297, 0.250663 };

	std::vector<double> weights10{ -0.459969 };
	std::vector<double> weights11{ 2.01929 };
	std::vector<double> weights12{ -2.0281 };
	std::vector<double> weights13{ 0.403286 };
	std::vector<double> weights14{ -0.807376 };

	nn.setWeight(0, 0, weights00);
	nn.setWeight(0, 1, weights01);
	nn.setWeight(0, 2, weights02);

	nn.setWeight(1, 0, weights10);
	nn.setWeight(1, 1, weights11);
	nn.setWeight(1, 2, weights12);
	nn.setWeight(1, 3, weights13);
	nn.setWeight(1, 4, weights14);

	std::vector<double> inputValues{ 0.0, 0.0 };
	nn.forwardProp(inputValues);

	std::vector<double> outputValues;
	nn.getOutputs(outputValues);
	std::cout << "Answer: " << outputValues[0] << std::endl;

	getchar();

	return 0;
}