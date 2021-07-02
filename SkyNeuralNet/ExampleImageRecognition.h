#pragma once

#include <iostream>

#include "external\stb_image\stb_image.h"
#include "NeuralNetwork/NeuralNet.h"
#include "Trainer.h"

void startExampleImageRecognition(bool printOnlyLastPass)
{
	// Neural network for recognizing images of numbers
	std::vector<unsigned int> neuronsPerLayer{ 784, 100, 10 };
	NeuralNet nn(neuronsPerLayer);
	// nn.setUseGPU(false);

	const int NUM_TRAINING_SETS = 5000;
	Trainer trainer(NUM_TRAINING_SETS);

	std::vector<double> inputValues;
	std::vector<double> expectedOutputForBackprop;
	std::vector<double> outputValues;

	// Variables for statistics
	const int KEEP_TRACK_NUM_LAST_CORRECT = 100;
	int trainingPass = 0;
	int numCorrect = 0;
	int numLastCorrect = 0;
	std::vector<bool> lastCorrect;
	lastCorrect.reserve(100);

	// Track time of overall training
	long startTime = std::clock();
	while (trainingPass < NUM_TRAINING_SETS)
	{
		// Track time of training pass
		long startTrainingPassTime = std::clock();

		// Load and read
		if (!trainer.loadImgOfNumber(trainingPass))
		{
			std::cout << "COULD NOT LOAD IMAGE" << std::endl;

			break;
		}
		trainingPass++;

		// Read input and expected output
		inputValues = trainer.getImgAsVector();
		expectedOutputForBackprop = trainer.getImgAnswer();

		// Forward propagation
		nn.forwardProp(inputValues);

		// Read output
		nn.getOutputs(outputValues);
		std::string answer = trainer.getAnswer(outputValues);
		std::string expected = trainer.getAnswer(expectedOutputForBackprop);

		// Keep track of the last 100 answers
		if (trainingPass >= KEEP_TRACK_NUM_LAST_CORRECT)
		{
			numLastCorrect -= lastCorrect[0];

			lastCorrect.erase(lastCorrect.begin());
		}

		// Right/wrong?
		if (answer[0] == expected[0])
		{
			numCorrect++;
			numLastCorrect++;
			lastCorrect.push_back(true);
		}
		else
			lastCorrect.push_back(false);

		// Print as one single string for faster output
		if ((printOnlyLastPass && trainingPass == NUM_TRAINING_SETS) ||
			!printOnlyLastPass)
		{
			std::cout << "Training pass: " + std::to_string(trainingPass) + "\n" +
				"Answer: " + answer + "\n" +
				"Expected: " + expected + "\n" +
				"Error: " + std::to_string(nn.calcError(expectedOutputForBackprop)) + "\n" +
				"CorrectTimes: " + std::to_string(numCorrect) + "\n" +
				"NumLastCorrect: " + std::to_string(numLastCorrect) + "\n";
		}

		// Back propagation
		nn.backProp(expectedOutputForBackprop);

		// Save and write time for this training pass
		long endTrainingPassTime = std::clock();

		if (!printOnlyLastPass)
		{
			std::cout << "Time for pass: " +
				std::to_string(endTrainingPassTime - startTrainingPassTime) +
				" milliseconds" + "\n\n";
		}
	}
	nn.endTrainingSession();

	// Save time for overall training
	long endTime = std::clock();
	long totalTime = (endTime - startTime);
	long seconds = totalTime / 1000;

	// Write time
	std::cout << "Done" << std::endl <<
		"Time: " << seconds / 60 << " minutes "
		<< seconds % 60 << " seconds" << " (" << totalTime
		<< " milliseconds)" << std::endl;

	// Output topology, weights and biases to file
	nn.outputNeuralNetToFile("E:/JavaProjs/SkyNeuralNetUsageExample/SkyNeuralNetUsage/android/assets/SkyNeuralNetSettings.ini");
}

void startExampleImageRecognitionTestBench()
{
	// Neural network for recognizing images of numbers
	std::vector<unsigned int> neuronsPerLayer{ 784, 100, 10 };
	NeuralNet nn(neuronsPerLayer);
	// nn.setUseGPU(false);

	// Best so far: 
	// (ETA: 0.05, ALPHA: 0.15, 51 000 correct, 91 correct last 100)
	for (float eta = 0.03f; eta <= 0.055f; eta += 0.01f)
	{
		for (float alpha = 0.05f; alpha <= 0.16f; alpha += 0.05f)
		{
			std::cout << "Configuration: (ETA: " + std::to_string(eta) +
				"  ALPHA: " + std::to_string(alpha) + ")\n";

			Neuron::setETA(eta);
			Neuron::setALPHA(alpha);

			nn.resetNetStructure();

			const int NUM_TRAINING_SETS = 60000;
			Trainer trainer(NUM_TRAINING_SETS);

			std::vector<double> inputValues;
			std::vector<double> expectedOutputForBackprop;
			std::vector<double> outputValues;

			// Variables for statistics
			const int KEEP_TRACK_NUM_LAST_CORRECT = 100;
			int trainingPass = 0;
			int numCorrect = 0;
			int numLastCorrect = 0;
			std::vector<bool> lastCorrect;
			lastCorrect.reserve(100);

			// Track time of overall training
			long startTime = std::clock();
			while (trainingPass < NUM_TRAINING_SETS)
			{
				// Track time of training pass
				long startTrainingPassTime = std::clock();

				// Load and read
				if (!trainer.loadImgOfNumber(trainingPass))
				{
					std::cout << "COULD NOT LOAD IMAGE" << std::endl;

					break;
				}
				trainingPass++;

				// Read input and expected output
				inputValues = trainer.getImgAsVector();
				expectedOutputForBackprop = trainer.getImgAnswer();

				// Forward propagation
				nn.forwardProp(inputValues);

				// Read output
				nn.getOutputs(outputValues);
				std::string answer = trainer.getAnswer(outputValues);
				std::string expected = trainer.getAnswer(expectedOutputForBackprop);

				// Keep track of the last 100 answers
				if (trainingPass >= KEEP_TRACK_NUM_LAST_CORRECT)
				{
					numLastCorrect -= lastCorrect[0];

					lastCorrect.erase(lastCorrect.begin());
				}

				// Right/wrong?
				if (answer[0] == expected[0])
				{
					numCorrect++;
					numLastCorrect++;
					lastCorrect.push_back(true);
				}
				else
					lastCorrect.push_back(false);

				// Print as one single string for faster output
				if (trainingPass == NUM_TRAINING_SETS)
				{
					std::cout << "Training pass: " + std::to_string(trainingPass) + "\n" +
						"Answer: " + answer + "\n" +
						"Expected: " + expected + "\n" +
						"Error: " + std::to_string(nn.calcError(expectedOutputForBackprop)) + "\n" +
						"CorrectTimes: " + std::to_string(numCorrect) + "\n" +
						"NumLastCorrect: " + std::to_string(numLastCorrect) + "\n";
				}

				// Back propagation
				nn.backProp(expectedOutputForBackprop);

				// Save and write time for this training pass
				long endTrainingPassTime = std::clock();
			}
			nn.endTrainingSession();

			// Save time for overall training
			long endTime = std::clock();
			long totalTime = (endTime - startTime);
			long seconds = totalTime / 1000;

			// Write time
			std::cout << "Time: " << seconds / 60 << " minutes "
				<< seconds % 60 << " seconds" << " (" << totalTime
				<< " milliseconds)" << std::endl << std::endl;
		}
	}

	std::cout << "Done" << std::endl;
}