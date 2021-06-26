#include "Neuron.h"
#include "ActivationFunction.h"
#include "Layer.h"

const float Neuron::ETA = 0.15;
const float Neuron::ALPHA = 0.5;

void Neuron::calcGradient(double delta)
{
	this->gradient = delta * ActivationFunction::derivative(this->outputValue);
}

double Neuron::sumWeightGradients(const std::vector<Neuron*>& nextLayerNeurons) const
{
	double sum = 0.0;

	for (int i = 0; i < this->outputWeights.size(); ++i)
		sum += this->outputWeights[i] * nextLayerNeurons[i]->getGradient();

	return sum;
}

Neuron::Neuron(double initialOutputValue, unsigned int numOutputWeights, 
	int myIndex)
	: outputValue(initialOutputValue), gradient(0.0), myIndex(myIndex)
{
	// Create random output weights
	for (int i = 0; i < numOutputWeights; ++i)
	{
		this->outputWeights.push_back((double)rand() / RAND_MAX);
		this->outputDeltaWeights.push_back(0.0);
	}
}

Neuron::~Neuron() { }

void Neuron::calcHiddenGradient(const std::vector<Neuron*>& nextLayerNeurons)
{
	double deltaOfWeights = this->sumWeightGradients(nextLayerNeurons);

	this->calcGradient(deltaOfWeights);
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

double Neuron::getOutputValue() const { return this->outputValue; }
double Neuron::getOutputWeight(int index) const { return this->outputWeights[index]; }
double Neuron::getGradient() const { return this->gradient; }

std::vector<double>& Neuron::getWeights() { return this->outputWeights; }
std::vector<double>& Neuron::getDeltaWeights() { return this->outputDeltaWeights; }
