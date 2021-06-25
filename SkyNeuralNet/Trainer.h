#pragma once

#include <fstream>
#include <string>
#include <vector>

class Trainer
{
private:
	std::ifstream readableDataFile;

public:
	Trainer();
	~Trainer();

	void loadFile(std::string path);
	void readLine(std::vector<std::string>& foundWords);
};