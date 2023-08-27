#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mnist.hh"

MNIST::MNIST(size_t batch_size, size_t number_of_batches, const std::string& filename)
    : batch_size(batch_size), number_of_batches(number_of_batches)
{
    // Ensure that batch size * number of batches doesn't exceed the available data points
    size_t total_data_points = batch_size * number_of_batches;
    if (total_data_points > 60000) 
    {
        std::cerr << "Error: The product of batch size and number of batches exceeds the available MNIST data points (60000)." << std::endl;
        return;
    }

    std::vector<std::vector<float>> data = parseMNISTCSV(filename);

    for (int i = 0; i < number_of_batches; ++i) 
    {
        // Create matrices for batches and targets
        batches.push_back(Matrix(Dimensions(batch_size, 784)));
        targets.push_back(Matrix(Dimensions(batch_size, 10)));
        
        // Allocate memory for matrices
        batches[i].allocateMemory();
        targets[i].allocateMemory();

        for (int k = 0; k < batch_size; k++)
        {
            for(int ind = 0; ind < 10; ind++)
            {
                if(data[i * batch_size + k][0] == ind)
                    targets[i][k + ind * batch_size] = 1.0f;
                else
                    targets[i][k + ind * batch_size] = 0.0f;
            }

            for (int j = 0; j < 784; j++)
            {
                batches[i][k + j * batch_size] = data[i * batch_size + k][j + 1];
            }
        }
        
        batches[i].copyHostToDevice();
        targets[i].copyHostToDevice();
    }

}

int MNIST::getNumOfBatches() 
{
    return number_of_batches;
}

std::vector<Matrix>& MNIST::getBatches() 
{
    return batches;
}

std::vector<Matrix>& MNIST::getTargets() 
{
    return targets;
}

std::vector<std::vector<float>> MNIST::parseMNISTCSV(const std::string& filename) 
{
    std::vector<std::vector<float>> data;

    std::ifstream file(filename);
    if (!file.is_open()) 
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return data;
    }

    // Skip the header line
    std::string header;
    std::getline(file, header);

    std::string line;
    while (std::getline(file, line)) 
    {
        std::vector<float> entry;
        std::istringstream iss(line);

        std::string value;
        std::getline(iss, value, ',');
        entry.push_back(static_cast<float>(std::stoi(value)));

        while (std::getline(iss, value, ',')) 
        {
            entry.push_back(std::stof(value) / 256.0f); 
        }

        data.push_back(entry);
    }

    return data;
}