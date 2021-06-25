#include "ActivationFunction.h"
#include <cmath>

double ActivationFunction::tanH(double x)
{
	return std::tanh(x);
}

double ActivationFunction::tanHDerivative(double x)
{
	return 1.0 - (x * x);
	//return 1.0 - (std::tanh(x) * std::tanh(x));
}

double ActivationFunction::function(double x)
{
	return ActivationFunction::tanH(x);
}

double ActivationFunction::derivative(double x)
{
	return ActivationFunction::tanHDerivative(x);
}
