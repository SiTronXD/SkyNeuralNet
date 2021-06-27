#include <iostream>

#include "external\stb_image\stb_image.h"
#include "NeuralNet.h"
#include "Trainer.h"

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

	Trainer trainer;
	trainer.loadImgOfNumber(0);
	std::vector<double> img = trainer.getImgAsVector();

	for (int i = 0; i < img.size(); ++i)
		std::cout << "i: " << i << " - " << img[i] << std::endl;

	for(int i = 0; i < 10; ++i)
		std::cout << trainer.getImgAnswer()[i] << std::endl;

	getchar();

	return 0;
}