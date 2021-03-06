#pragma once

#include <vector>

class Layer;

class Neuron
{
private:
	static float ETA;		// Overall net learning rate [0.0 .. 1.0]
	static float ALPHA;	// Momentum, multiplier of last deltaWeight, [0.0 .. n]

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
	void setGradient(double gradientValue);
	void setWeight(int index, double newWeightValue, double newDeltaWeightValue);
	void updateWeights(Layer* previousLayer);

	static void setETA(float newETA);
	static void setALPHA(float newALPHA);

	inline const double& getOutputValue() const { return this->outputValue; }
	inline const double& getOutputWeight(int index) const { return this->outputWeights[index]; }
	inline const double& getGradient() const { return this->gradient; }

	inline std::vector<double>& getWeights() { return this->outputWeights; };
	inline std::vector<double>& getDeltaWeights() { return this->outputDeltaWeights; }

	inline static const float& getETA() { return ETA; }
	inline static const float& getALPHA() { return ALPHA; }
};