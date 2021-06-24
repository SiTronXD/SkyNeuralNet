#include "NeuralNet.h"

NeuralNet::NeuralNet(std::vector<unsigned int> neuronPerLayer)
{
	for (int i = 0; i < neuronPerLayer.size(); ++i)
	{
		// Get number of output weights, if it exists
		int numOutputWeights = 0;
		if (i < neuronPerLayer.size() - 1)
			numOutputWeights = neuronPerLayer[i + 1];

		// Get previous layer, if it exists
		Layer* previousLayer = nullptr;
		if (i > 0)
			previousLayer = &this->layers[i - 1];

		// Add layer
		this->layers.push_back(
			Layer(
				neuronPerLayer[i], 
				numOutputWeights, 
				numOutputWeights != 0, 
				previousLayer
			)
		);
	}
}

NeuralNet::~NeuralNet()
{
}
