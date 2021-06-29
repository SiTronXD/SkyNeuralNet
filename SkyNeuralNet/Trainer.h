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

	inline const std::vector<double>& getImgAsVector() const 
		{ return this->trainingData.getImgAsVector(); }
	inline const std::vector<double>& getImgAnswer() const 
		{ return this->imgAnswer; }
};