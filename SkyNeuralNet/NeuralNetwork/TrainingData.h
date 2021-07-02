#pragma once

#include <fstream>
#include <string>
#include <vector>

class TrainingData
{
private:
	// Load file
	std::ifstream readableDataFile;

	// Load image
	int imageWidth;
	int imageHeight;
	int imageComponents;

	unsigned char* imageData;

	std::vector<double> imgVec;

public:
	TrainingData();
	~TrainingData();

	const bool loadFile(std::string path);
	const bool loadImg(std::string path);

	void readLine(std::vector<std::string>& foundWords);

	inline bool doesFileExist(const std::string& name)
	{
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);
	}

	inline const std::vector<double>& getImgAsVector() const 
		{ return this->imgVec; }
};