#pragma once

#include <vector>

class Layer;

class Neuron
{
private:
	static const float ETA;		// Overall net learning rate [0.0 .. 1.0]
	static const float ALPHA;	// Momentum, multiplier of last deltaWeight, [0.0 .. n]

	int myIndex;

	double outputValue;
	double gradient;

	double(*activationFunctionDerivative)(double);

	std::vector<double> outputWeights;
	std::vector<double> outputDeltaWeights;

	void calcGradient(double delta);

	double sumWeightGradients(const std::vector<Neuron*>& nextLayerNeurons) const;

public:
	Neuron(double initialOutputValue, unsigned int numOutputWeights, int myIndex,
		bool isOutputNeuron);
	~Neuron();

	void calcHiddenGradient(const std::vector<Neuron*>& nextLayerNeurons);
	void calcOutputGradient(double targetValue);
	void setOutputValue(double outputValue);
	void updateWeights(Layer* previousLayer);

	double getOutputValue() const;
	double getOutputWeight(int index) const;
	double getGradient() const;

	std::vector<double>& getWeights();
	std::vector<double>& getDeltaWeights();
};