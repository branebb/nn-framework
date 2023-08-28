#include "coordinates_test.hh"
#include <random>

CoordinatesDataset::CoordinatesDataset(size_t batch_size, size_t number_of_batches) :
    batch_size(batch_size), number_of_batches(number_of_batches)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < number_of_batches; i++) 
    {
        batches.push_back(Matrix(Dimensions(batch_size, 2)));
        targets.push_back(Matrix(Dimensions(batch_size, 2)));
        
        batches[i].allocateMemory();
        targets[i].allocateMemory();

        for (int k = 0; k < batch_size; k++) {
            batches[i][k] = dist(rng);
            batches[i][batches[i].dims.x + k] = dist(rng);

            if ((batches[i][k] > 0 && batches[i][batches[i].dims.x + k] > 0)) {
                targets[i][k] = 0;
				targets[i][batches[i].dims.x + k] = 1;

            } else {
                targets[i][k] = 1;
				targets[i][batches[i].dims.x + k] = 0;
            }
        }

        batches[i].copyHostToDevice();
        targets[i].copyHostToDevice();
    }
}


int CoordinatesDataset::getNumOfBatches() {
	return number_of_batches;
}

std::vector<Matrix>& CoordinatesDataset::getBatches() {
	return batches;
}

std::vector<Matrix>& CoordinatesDataset::getTargets() {
	return targets;
}