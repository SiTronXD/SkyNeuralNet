#pragma once

class ActivationFunction
{
public:
	static double sigmoid(double x);
	static double sigmoidDerivative(double x);

	static double tanH(double x);
	static double tanHDerivative(double x);

	static double relu(double x);
	static double reluDerivative(double x);


	static double activateOutput(double x);
	static double activateHidden(double x);

	static double outputDerivative(double x);
	static double hiddenDerivative(double x);
};