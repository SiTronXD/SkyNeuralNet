#pragma once

#include "Layer.h"

class NeuralNet
{
private:
	std::vector<Layer> layers;

public:
	NeuralNet(std::vector<unsigned int> neuronPerLayer);
	~NeuralNet();
};