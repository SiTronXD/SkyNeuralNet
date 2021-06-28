#pragma once

#include "NeuralNetwork\TrainingData.h"

class Trainer
{
private:
	TrainingData trainingData;

	int* setNum;

	std::vector<double> imgAnswer;

public:
	Trainer(int numTrainingSets);
	~Trainer();

	const bool loadImgOfNumber(int imgIndex);

	const std::vector<double>& getImgAsVector() const;
	const std::vector<double>& getImgAnswer() const;
};