#include "Trainer.h"
#include <sstream>

Trainer::Trainer()
{
}

Trainer::~Trainer()
{
}

void Trainer::loadFile(std::string path)
{
	this->readableDataFile.open(path);
}

void Trainer::readLine(std::vector<std::string>& foundWords)
{
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
