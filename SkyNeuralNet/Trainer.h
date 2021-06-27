#pragma once

#include <fstream>
#include <string>
#include <vector>

class Trainer
{
private:
	// Load file
	std::ifstream readableDataFile;

	// Load image
	int imageWidth;
	int imageHeight;
	int imageComponents;

	unsigned char* imageData;

	std::vector<double> imgArray;
	std::vector<double> imgAnswer;

public:
	Trainer();
	~Trainer();

	bool loadFile(std::string path);
	bool loadImg(std::string path);
	bool loadImgOfNumber(int imgIndex);

	void readLine(std::vector<std::string>& foundWords);

	const std::vector<double>& getImgAsVector() const;
	const std::vector<double>& getImgAnswer() const;
};