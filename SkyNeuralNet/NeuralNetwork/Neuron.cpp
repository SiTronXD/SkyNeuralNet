#include "Neuron.h"
#include "ActivationFunction.h"
#include "Layer.h"

// Best for XOR example: (ETA: 0.15, ALPHA: 0.5)

float Neuron::ETA = 0.04;
float Neuron::ALPHA = 0.15;

// ETA: 0.01, ALPHA: 0.35, 5000 sets, Last 100 correct: 84, Correct: 2815
// ETA: 0.02, ALPHA: 0.35, 5000 sets, Last 100 correct: 85, Correct: 3490
// ETA: 0.05, ALPHA: 0.35, 5000 sets, Last 100 correct: 86, Correct: 3734
// ETA: 0.10, ALPHA: 0.35, 5000 sets, Last 100 correct: 78, Correct: 3645
// ETA: 0.15, ALPHA: 0.35, 5000 sets, Last 100 correct: 71, Correct: 3431
// ETA: 0.25, ALPHA: 0.35, 5000 sets, Last 100 correct: 47, Correct: 2363
// ETA: 0.35, ALPHA: 0.35, 5000 sets, Last 100 correct: 40, Correct: 1516

void Neuron::calcGradient(double delta)
{
	this->gradient = delta * this->activationFunctionDerivative(this->outputValue);
}

double Neuron::sumWeightGradients(const std::vector<Neuron*>& nextLayerNeurons) const
{
	double sum = 0.0;

	for (int i = 0; i < this->outputWeights.size(); ++i)
		sum += this->outputWeights[i] * nextLayerNeurons[i]->getGradient();

	return sum;
}

Neuron::Neuron(double initialOutputValue, unsigned int numOutputWeights, 
	int myIndex, bool isOutputNeuron)
	: outputValue(initialOutputValue), gradient(0.0), myIndex(myIndex)
{
	// Create random output weights
	for (int i = 0; i < numOutputWeights; ++i)
	{
		this->outputWeights.push_back((double)rand() / RAND_MAX);
		this->outputDeltaWeights.push_back(0.0);
	}

	// Choose derivative for activation function
	if (isOutputNeuron)
		this->activationFunctionDerivative = ActivationFunction::outputDerivative;
	else
		this->activationFunctionDerivative = ActivationFunction::hiddenDerivative;
}

Neuron::~Neuron() { }

void Neuron::calcHiddenGradient(const std::vector<Neuron*>& nextLayerNeurons)
{
	double swg = this->sumWeightGradients(nextLayerNeurons);

	this->calcGradient(swg);
}

void Neuron::calcOutputGradient(double targetValue)
{
	double delta = targetValue - this->outputValue;

	this->calcGradient(delta);
}

void Neuron::setOutputValue(double outputValue)
{
	this->outputValue = outputValue;
}

void Neuron::setGradient(double gradientValue)
{
	this->gradient = gradientValue;
}

void Neuron::setWeight(
	int index, 
	double newWeightValue, 
	double newDeltaWeightValue
)
{
	this->outputWeights[index] = newWeightValue;
	this->outputDeltaWeights[index] = newDeltaWeightValue;
}

void Neuron::updateWeights(Layer* nextLayer)
{
	std::vector<Neuron*>& nextNeurons = nextLayer->getNeurons();

	// Go through and update all weights
	for (int i = 0; i < this->outputWeights.size(); ++i)
	{
		double oldDeltaWeight = this->outputDeltaWeights[i];

		double newDeltaWeight =
			// Individual input, magnified by the gradient and train rate:
			this->ETA * this->outputValue * nextNeurons[i]->getGradient() +
			// Also add momentum: a fraction of the previous delta weight
			this->ALPHA * oldDeltaWeight;

		this->outputDeltaWeights[i] = newDeltaWeight;
		this->outputWeights[i] += newDeltaWeight;
	}
}

void Neuron::setETA(float newETA) { Neuron::ETA = newETA; }
void Neuron::setALPHA(float newALPHA) { Neuron::ALPHA = newALPHA; }
