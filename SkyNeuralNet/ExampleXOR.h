#pragma once

#include <iostream>

#include "NeuralNetwork/NeuralNet.h"
#include "Trainer.h"

void startExampleXOR()
{
	// XOR neural network
	std::vector<unsigned int> neuronsPerLayer{ 2, 4, 1 };
	NeuralNet nn(neuronsPerLayer);
	//nn.setUseGPU(false);

	Trainer trainer;
	trainer.loadFile("XORTrainingData.txt");

	std::vector<std::string> readValues;
	trainer.readLine(readValues);
	trainer.readLine(readValues);

	int trainingPass = 1;
	while (readValues.size() > 0)
	{
		double input0 = std::stod(readValues[1]);
		double input1 = std::stod(readValues[2]);
		trainer.readLine(readValues);
		double expectedOutput = std::stod(readValues[1]);

		std::vector<double> inputValues{ input0, input1 };
		std::vector<double> expectedOutputForBackprop{ expectedOutput };
		std::vector<double> outputValues;

		std::cout << "Training pass: " << trainingPass << std::endl;
		trainingPass++;

		// Forward prop
		nn.forwardProp(inputValues);
		std::cout << "Input: " << input0 << " " << input1 << std::endl;

		// Read output
		nn.getOutputs(outputValues);
		std::cout << "Answer: " << outputValues[0] << std::endl;
		std::cout << "Error: " << nn.calcError(expectedOutputForBackprop) << std::endl;
		std::cout << std::endl;

		// Train
		nn.backProp(expectedOutputForBackprop);

		trainer.readLine(readValues);
	}
	nn.endTrainingSession();
}