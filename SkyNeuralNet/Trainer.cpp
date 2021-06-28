#include "Trainer.h"
#include <iostream>

Trainer::Trainer(int numTrainingSets)
	: setNum(nullptr)
{
	// Load number order
	this->setNum = new int[numTrainingSets];
	this->trainingData.loadFile("E:/simon/Bilder - Data Drive/TrainingSets/mnist_train/TrainingSetNumberOrder.txt");

	// Insert number order
	std::vector<std::string> readWords;
	for (int i = 0; i < numTrainingSets; ++i)
	{
		this->trainingData.readLine(readWords);

		this->setNum[i] = std::stoi(readWords[0]);
	}
}

Trainer::~Trainer()
{
	delete[] this->setNum;
}

// Training set is from MNIST database 
// (converted to .png from this repository: 
// https://github.com/pjreddie/mnist-csv-png)
const bool Trainer::loadImgOfNumber(int imgIndex)
{
	// Example: "mnist_train/train/000000-num5.png"
	std::string name = "E:/simon/Bilder - Data Drive/TrainingSets/mnist_train/train/";
	int digits = imgIndex == 0 ? 1 : log10(imgIndex) + 1;
	for (int i = 6 - digits; i > 0; --i)
		name += "0";
	name += std::to_string(imgIndex);

	// Create file name
	std::string tempName = 
		name + "-num" + std::to_string(this->setNum[imgIndex]) + ".png";

	// Found file
	if (this->trainingData.doesFileExist(tempName))
	{
		this->trainingData.loadImg(tempName);

		// Create answer vector
		this->imgAnswer.clear();
		for (int j = 0; j < 10; ++j)
			this->imgAnswer.push_back(j == this->setNum[imgIndex] ? 1.0 : 0.0);

		return true;
	}

	// File was not found...
	std::cout << "Can not find training image..." << std::endl;

	return false;
}

const std::vector<double>& Trainer::getImgAsVector() const
{
	return this->trainingData.getImgAsVector();
}

const std::vector<double>& Trainer::getImgAnswer() const
{
	return this->imgAnswer;
}