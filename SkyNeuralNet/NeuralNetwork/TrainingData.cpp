#include "TrainingData.h"
#include <iostream>
#include <sstream>
#include "../external/stb_image/stb_image.h"

TrainingData::TrainingData()
	: imageWidth(0), imageHeight(0), imageComponents(0), imageData(0)
{
	// Allocate array once at the start
	this->imgVec.reserve(this->imageWidth * this->imageHeight);
}

TrainingData::~TrainingData()
{ 
	stbi_image_free(this->imageData);
}

const bool TrainingData::loadFile(std::string path)
{
	this->readableDataFile.open(path);

	return this->readableDataFile.is_open();
}

const bool TrainingData::loadImg(std::string path)
{
	// Free old image
	stbi_image_free(this->imageData);
	this->imageData = nullptr;

	// Load image and assign variables
	this->imageData = stbi_load(
		path.c_str(), 
		&this->imageWidth, 
		&this->imageHeight, 
		&this->imageComponents, 
		4
	);

	// Could not load image
	if (this->imageData == NULL)
	{
		return false;
	}

	unsigned char* pixelOffset;
	double r, g, b;

	// Load grey-scale values
	this->imgVec.clear();
	for (int y = 0; y < this->imageHeight; ++y)
	{
		for (int x = 0; x < this->imageWidth; ++x)
		{
			pixelOffset = imageData + (x + this->imageWidth * y) * 4;
			r = (double) pixelOffset[0] / 255.0;
			g = (double) pixelOffset[1] / 255.0;
			b = (double) pixelOffset[2] / 255.0;

			this->imgVec.push_back((r + g + b) / 3.0);
		}
	}

	return true;
}

void TrainingData::readLine(std::vector<std::string>& foundWords)
{
	// Remove previous words
	foundWords.clear();

	// Read line from file
	std::string line;
	std::getline(this->readableDataFile, line);

	// Add words split by spaces to foundWords
	std::stringstream ss(line);
	std::string word;
	while (ss >> word)
		foundWords.push_back(word);
}