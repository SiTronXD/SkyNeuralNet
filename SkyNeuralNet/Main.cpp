
#include "ExampleXOR.h"
#include "ExampleImageRecognition.h"

int main()
{
	// Catch memory leaks
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	std::cout << "Started training...\n";

	// startExampleXOR();
	// startExampleImageRecognition(false);

	// Test bench for different parameters
	for (float eta = 0.05f; eta <= 0.05f; eta += 0.05f)
	{
		for (float alpha = 0.20; alpha <= 0.25; alpha += 0.05f)
		{
			std::cout << "Configuration: (ETA: "  + std::to_string(eta) +
				"  ALPHA: " + std::to_string(alpha) + ")\n";

			Neuron::setETA(eta);
			Neuron::setALPHA(alpha);

			startExampleImageRecognition(true);
		}
	}

	cudaDeviceReset();

	getchar();

	return 0;
}