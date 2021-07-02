
#include "ExampleXOR.h"
#include "ExampleImageRecognition.h"

int main()
{
	// Catch memory leaks
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	std::cout << "Started training...\n";

	// startExampleXOR();
	// startExampleImageRecognition(false);
	startExampleImageRecognitionTestBench();
	
	getchar();

	return 0;
}