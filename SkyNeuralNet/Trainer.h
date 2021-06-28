#pragma once

#include "NeuralNetwork\TrainingData.h"

class Trainer
{
private:
	TrainingData trainingData;

	int* setNum;

	std::vector<double> imgAnswer;

public:
	Trainer();
	Trainer(int numTrainingSets);
	~Trainer();

	const bool loadFile(std::string filePath);
	const bool loadImgOfNumber(int imgIndex);

	void readLine(std::vector<std::string>& foundWords);

	const std::vector<double>& getImgAsVector() const;
	const std::vector<double>& getImgAnswer() const;
};