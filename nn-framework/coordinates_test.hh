#pragma once

#include <vector>

#include "nn-framework/headers/structures/matrix.hh"

class CoordinatesDataset 
{
private:
	size_t batch_size;
	size_t number_of_batches;

	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

public:

	CoordinatesDataset(size_t batch_size, size_t number_of_batches);

	int getNumOfBatches();
	std::vector<Matrix>& getBatches();
	std::vector<Matrix>& getTargets();

};