#include <vector>
#include <string>

#include "nn-framework/headers/structures/matrix.hh"

class MNIST 
{
private:
	size_t batch_size;
	size_t number_of_batches;

	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

public:

	MNIST(size_t batch_size, size_t number_of_batches, const std::string& filename);

	int getNumOfBatches();

	std::vector<Matrix>& getBatches();
	std::vector<Matrix>& getTargets();

    std::vector<std::vector<float>> parseMNISTCSV(const std::string& filename);
};