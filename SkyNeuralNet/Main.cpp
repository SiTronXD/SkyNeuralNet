#include <iostream>

#include "external\stb_image\stb_image.h"
#include "NeuralNet.h"
#include "Trainer.h"

std::string getAnswer(std::vector<double> answers)
{
	int currentIndex = -1;
	double currentBest = 0.0;

	for (int i = 0; i < answers.size(); ++i)
	{
		if (answers[i] > currentBest)
		{
			currentIndex = i;
			currentBest = answers[i];
		}
	}

	return std::to_string(currentIndex) + ": " + std::to_string(currentBest * 100.0) + "%";
}

int main()
{
	// Catch memory leaks
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	/*std::vector<unsigned int> neuronsPerLayer{ 2, 4, 1 };
	NeuralNet nn(neuronsPerLayer);

	Trainer trainer;
	trainer.loadFile("TrainingData.txt");

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
		std::cout << "Error: " << nn.getError(expectedOutputForBackprop) << std::endl;
		std::cout << std::endl;

		// Train
		nn.backProp(expectedOutputForBackprop);
			
		trainer.readLine(readValues);
	}*/

	std::vector<unsigned int> neuronsPerLayer{ 784, 100, 10 };
	NeuralNet nn(neuronsPerLayer);

	Trainer trainer;

	std::vector<double> inputValues;
	std::vector<double> expectedOutputForBackprop;
	std::vector<double> outputValues;

	int trainingPass = 1;
	int numCorrect = 0;
	std::vector<bool> lastCorrect;
	while (trainingPass < 5000)
	{
		// Load and read
		if (!trainer.loadImgOfNumber(trainingPass - 1))
		{
			std::cout << "COULD NOT LOAD IMAGE" << std::endl;

			break;
		}
		inputValues = trainer.getImgAsVector();
		expectedOutputForBackprop = trainer.getImgAnswer();

		std::cout << "Training pass: " << trainingPass << std::endl;
		trainingPass++;

		// Forward prop
		nn.forwardProp(inputValues);

		// Read output
		nn.getOutputs(outputValues);
		outputValues.pop_back();
		std::string answer = getAnswer(outputValues);
		std::string expected = getAnswer(expectedOutputForBackprop);

		if (trainingPass >= 100)
			lastCorrect.erase(lastCorrect.begin());

		if (answer[0] == expected[0])
		{
			numCorrect++;
			lastCorrect.push_back(true);
		}
		else
			lastCorrect.push_back(false);

		int numLastCorrect = 0;
		for (int i = 0; i < lastCorrect.size(); ++i)
		{
			if (lastCorrect[i])
				numLastCorrect++;
		}

		std::cout << "Answer: " << answer << std::endl;
		std::cout << "Expected: " << expected << std::endl;
		std::cout << "Error: " << nn.getError(expectedOutputForBackprop) << std::endl;
		std::cout << "CorrectTimes: " << numCorrect << std::endl;
		std::cout << "NumLastCorrect: " << numLastCorrect << std::endl;
		std::cout << std::endl;

		// Train
		nn.backProp(expectedOutputForBackprop);
	}

	getchar();

	return 0;
}