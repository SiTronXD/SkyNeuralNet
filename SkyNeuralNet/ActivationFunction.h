#pragma once

class ActivationFunction
{
private:
	static double tanH(double x);
	static double tanHDerivative(double x);

public:
	static double function(double x);
	static double derivative(double x);
};