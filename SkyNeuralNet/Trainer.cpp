#include "Trainer.h"
#include <iostream>
#include <sstream>
#include "external/stb_image/stb_image.h"

Trainer::Trainer() 
	: imageWidth(0), imageHeight(0), imageComponents(0), imageData(0)
{ 

}

Trainer::~Trainer() 
{ 
	stbi_image_free(this->imageData);
}

inline bool doesFileExist(const std::string& name)
{
	struct stat buffer;

	return (stat(name.c_str(), &buffer) == 0);
}

bool Trainer::loadFile(std::string path)
{
	this->readableDataFile.open(path);

	return this->readableDataFile.is_open();
}

bool Trainer::loadImg(std::string path)
{
	// Free old image
	stbi_image_free(this->imageData);

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
	double r, g, b, a;

	// Load grey-scale values
	this->imgArray.clear();
	for (int y = 0; y < this->imageHeight; ++y)
	{
		for (int x = 0; x < this->imageWidth; ++x)
		{
			pixelOffset = imageData + (x + this->imageWidth * y) * 4;
			r = (double) pixelOffset[0] / 255.0;
			g = (double) pixelOffset[1] / 255.0;
			b = (double) pixelOffset[2] / 255.0;

			this->imgArray.push_back((r + g + b) / 3.0);
		}
	}

	return true;
}

// Training set is from MNIST database 
// (converted to .png from this repository: 
// https://github.com/pjreddie/mnist-csv-png)
bool Trainer::loadImgOfNumber(int imgIndex)
{
	// Example: "mnist_train/train/000000-num5.png"
	std::string name = "E:/simon/Bilder - Data Drive/TrainingSets/mnist_train/train/";
	std::string tempName;
	int l = imgIndex == 0 ? 1 : log10(imgIndex) + 1;
	for (int i = 6 - l; i > 0; --i)
	{
		name += "0";
	}
	name += std::to_string(imgIndex);

	for (int i = 0; i < 10; ++i)
	{
		tempName = name + "-num" + std::to_string(i) + ".png";

		// Found file
		if (doesFileExist(tempName))
		{
			this->loadImg(tempName);

			// Create answer
			this->imgAnswer.clear();
			for (int j = 0; j < 10; ++j)
				this->imgAnswer.push_back(j == i ? 1.0 : 0.0);

			return true;
		}
	}

	// File was not found...
	std::cout << "Can not find training image..." << std::endl;

	return false;
}

void Trainer::readLine(std::vector<std::string>& foundWords)
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

const std::vector<double>& Trainer::getImgAsVector() const
{
	return this->imgArray;
}

const std::vector<double>& Trainer::getImgAnswer() const
{
	return this->imgAnswer;
}
