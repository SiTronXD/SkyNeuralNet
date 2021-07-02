#include "Trainer.h"
#include <iostream>

Trainer::Trainer()
	: setNum(nullptr)
{
}

Trainer::Trainer(int numTrainingSets)
	: setNum(nullptr)
{
	// Load number order
	this->setNum = new int[numTrainingSets];
	if (!this->trainingData.loadFile("E:/simon/Bilder - Data Drive/TrainingSets/mnist_train/TrainingSetNumberOrder.txt"))
	{
		std::cout << "Could not find number order text file..." << std::endl;

		return;
	}

	// Insert number order
	std::vector<std::string> readWords;
	for (int i = 0; i < numTrainingSets; ++i)
	{
		this->trainingData.readLine(readWords);

		this->setNum[i] = std::stoi(readWords[0]);
	}

	// Init img vector
	this->imgAnswer.resize(10, 0.0);
}

Trainer::~Trainer()
{
	delete[] this->setNum;
}

const bool Trainer::loadFile(std::string filePath)
{
	return this->trainingData.loadFile(filePath);
}

// Training set is from MNIST database 
// (converted to .png from this repository: 
// https://github.com/pjreddie/mnist-csv-png)
const bool Trainer::loadImgOfNumber(int imgIndex)
{
	// Example: "mnist_train/train/000000-num5.png"
	std::string name = "E:/simon/Bilder - Data Drive/TrainingSets/mnist_train/train/";
	int digits = imgIndex == 0 ? 1 : (int) log10(imgIndex) + 1;
	name.append(6 - digits, '0'); // Add zeros to the front of the file name
	name.append(std::to_string(imgIndex));
	name.append("-num");

	// Create file name
	std::string foundFileName = 
		name + std::to_string(this->setNum[imgIndex]) + ".png";

	// Found file
	if (this->trainingData.doesFileExist(foundFileName))
	{
		this->trainingData.loadImg(foundFileName);

		// Clear and set answer array
		memset(&this->imgAnswer[0], 0, this->imgAnswer.size() * sizeof(this->imgAnswer[0]));
		this->imgAnswer[this->setNum[imgIndex]] = 1.0;

		return true;
	}

	// File was not found...
	std::cout << "Can not find training image..." << std::endl;

	return false;
}

void Trainer::readLine(std::vector<std::string>& foundWords)
{
	this->trainingData.readLine(foundWords);
}

std::string Trainer::getAnswer(const std::vector<double>& answers)
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

	// <index>: <guess percentage>% 
	return std::to_string(currentIndex) + ": " + std::to_string(currentBest * 100.0) + "%";
}
