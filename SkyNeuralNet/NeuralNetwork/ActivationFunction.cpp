#include "ActivationFunction.h"
#include <cmath>

double ActivationFunction::sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double ActivationFunction::sigmoidDerivative(double x)
{
	double s = sigmoid(x);

	return s * (1.0 - s);
}

double ActivationFunction::tanH(double x)
{
	return std::tanh(x);
}

double ActivationFunction::tanHDerivative(double x)
{
	//return 1.0 - (x * x);
	return 1.0 - (std::tanh(x) * std::tanh(x));
}

double ActivationFunction::relu(double x)
{
	return std::fmax(0.0, x);
}

double ActivationFunction::reluDerivative(double x)
{
	return x >= 0.0 ? 1.0 : 0.0;
}


// ------ Activation functions for hidden/output layers ------

double ActivationFunction::activateOutput(double x)
{
	return ActivationFunction::sigmoid(x);
}

double ActivationFunction::activateHidden(double x)
{
	return ActivationFunction::relu(x);
}

double ActivationFunction::outputDerivative(double x)
{
	return ActivationFunction::sigmoidDerivative(x);
}

double ActivationFunction::hiddenDerivative(double x)
{
	return ActivationFunction::reluDerivative(x);
}
