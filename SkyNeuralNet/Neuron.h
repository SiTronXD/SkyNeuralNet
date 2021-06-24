#pragma once

#include <vector>

class Neuron
{
private:
	double outputValue;

	std::vector<double> outputWeights;

public:
	Neuron(double initialOutputValue, unsigned int numOutputWeights);
	~Neuron();

	void setOutputValue(double outputValue);

	double getOutputValue() const;
	double getOutputWeight(int index) const;

	std::vector<double>& getWeights();
};