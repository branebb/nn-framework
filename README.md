# CUDA Neural network framework

Welcome to the CUDA Neural Network Framework! This guide will help you set up and run the framework on your Windows system. Please follow the instructions below to ensure a smooth experience.

## Requirements

Before you start, make sure you have the following prerequisites installed on your system:
- Editor: You can choose between `Visual Studio Code` or `Microsoft Visual Studio`. While you can use only `Microsoft Visual Studio`, if you opt for `Visual Studio Code`, please ensure you also have `Microsoft Visual Studio` installed.
  
- CUDA Toolkit: Download and install the `CUDA Toolkit` on your system.
  
- Operating System: Currently, the framework supports only `Windows`.

## Running the Project
Follow these steps to build and run the project:
- Path Setup: Ensure that the paths to both `CUDA` and `Microsoft Visual Studio Code/bin` are added to your environment variables (`PATH`).
  
- Project Setup: After cloning or downloading the project, you should have the following items:
  - `nn-framework` folder
  - `CMakeLists.txt` file
  - `windows-build-run-debug.bat` file
- Configure MSBuild: Open the `windows-build-run-debug.bat` file and replace `"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe"` with the correct path to `MSBuild.exe` on your computer.
  
- Test the Setup: Open a terminal in `Visual Studio Code` and execute the following command: `.\windows-build-run-debug.bat`. If everything is configured correctly, your code should `build and run` without any issues.

## Components Overview
Before you start using the `CUDA Neural Network Framework`, take a moment to understand the key components and concepts that are utilized in the project. This will help you gain a clear understanding of how the framework operates.

### MNIST Dataset
The `MNIST dataset` is a widely used dataset for training and testing machine learning models. It consists of a large collection of handwritten digits along with their corresponding labels.
To use the `MNIST dataset` with the `CUDA Neural Network Framework`, follow these steps:
- Download the Dataset: Download the `MNIST dataset` from the [`Kaggle dataset page`](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
  
- Extract the Dataset: After downloading, extract the dataset files.
  
- Organize the Dataset: Place the extracted dataset folder in the `nn-framework` directory of your project.

With a clear understanding of the `MNIST dataset` and other key components, you're now ready to dive into using the `CUDA Neural Network Framework`.

### Layers
In the context of neural networks, layers play a crucial role in defining the architecture and behavior of the model. Each layer type serves a specific purpose in transforming input data and learning features. Here are the types of layers implemented in the CUDA Neural Network Framework:
#### Linear layer
The `Linear Layer`, also known as a `Fully Connected Layer`, connects each input neuron to every neuron in the following layer. It performs a linear transformation of the input data.

```math
Z = A \cdot W + b
```

This formula computes the weighted sum of the input matrix `A` with the weight matrix `W`, and adds the bias vector `b`. The resulting matrix `Z` forms the input for the subsequent activation layer. 

The `Linear Layer` is a fundamental building block in constructing neural network architectures, enabling the flow of information between layers.

### Activation Functions
#### Sigmoid 
The `Sigmoid activation` function is a smooth, S-shaped curve that maps input values to a range between 0 and 1. It's commonly used in binary classification tasks where the output needs to be interpreted as a probability.

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

#### Tanh
The `Hyperbolic Tangent (Tanh) activation` function is similar to the Sigmoid but maps input values to a range between -1 and 1. It's often used in recurrent neural networks due to its zero-centered output.

```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

#### ReLU
The `Rectified Linear Unit (ReLU) activation` function is widely used in neural networks. It replaces negative input values with zero and leaves positive values unchanged. ReLU is computationally efficient and addresses the vanishing gradient problem.

```math
\text{ReLU}(x) = \max(0, x)
```

#### Softmax
The `Softmax activation` function is used in the output layer of neural networks for multi-class classification. It converts raw scores into a probability distribution, with each class receiving a probability score between 0 and 1 that sums to 1.

```math
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
```

### Cost Functions
#### Cross-Entropy (CE)
`Cross-Entropy (CE`) is a versatile loss function used extensively in classification tasks. It measures the dissimilarity between predicted class probabilities and the true class labels. `CE` encourages the model to assign higher probabilities to the correct classes, driving accurate predictions.

For a dataset with `m` training examples and `K` classes, the Cross-Entropy cost `L` can be calculated as:

```math
L(y, \hat{y}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})
```

#### Mean Square Error (MSE)
The `Mean Squared Error (MSE)` cost function is widely used in regression tasks. It measures the average of the squared differences between the predicted values and the true target values. This loss function aims to minimize the squared deviations between predictions and targets, making it suitable for problems where the magnitude of errors matters.

For a dataset with `m` training examples, the MSE cost `L` can be calculated as:

```math
L(y, \hat{y}) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
```

It's important to choose the appropriate loss function based on the problem you're addressing and the desired behavior of your model. Both MSE and Cross-Entropy play crucial roles in training neural networks effectively and achieving accurate results.

### Optimizers
Optimizers play a crucial role in training neural networks by iteratively adjusting model parameters to minimize the loss function. 

#### Gradient descent
`Gradient Descent` is a fundamental optimization algorithm used to update model parameters in the direction that reduces the loss. It calculates the gradient of the loss with respect to the parameters and updates the parameters by subtracting a fraction of the gradient. The `learning rate (alpha)` determines the step size.

```math
\theta = \theta - \alpha \nabla J(\theta)
```
#### Adam
`Adam (Adaptive Moment Estimation)` is an advanced optimization algorithm that combines the benefits of both `Momentum` and `RMSProp`. It adapts the learning rate for each parameter by computing adaptive per-parameter learning rates. It also includes momentum-like behavior to overcome flat regions in the loss landscape.

```math
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \nabla J(\theta_t)
```
```math
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot (\nabla J(\theta_t))^2
```
```math
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} \cdot m_t
```

### Regularization
Regularization techniques are employed to prevent overfitting in neural networks. By adding penalty terms to the loss function, regularization discourages overly complex models and helps achieve better generalization performance on unseen data.
#### L2
`L2 regularization`, also known as `weight decay`, is a widely used technique to control the complexity of neural networks. It involves adding a penalty term proportional to the squared magnitude of the model's weights to the loss function. This encourages the network to prioritize smaller weight values, preventing individual weights from growing too large.

```math
L_{\text{regularized}} = L_{\text{original}} + \frac{\lambda}{2} \sum_{w} w^2
```

## Usage
### Initialization and Configuration
Start by initializing the necessary components for your neural network, including data loading, regularization, cost function, optimizer, learning rate, and architecture.
```cpp
float lambda = 0.00f;

float beta1 = 0.9f;
float beta2 = 0.999f;
float epsilon = 1e-8f;

float learning_rate = 0.01f;

MNIST traindata(32, 1875, "datasets/mnist_train.csv");

L2 L2(lambda);

CrossEntropyCost crossEntropy(&L2);

AdamOptimizer adam(beta1, beta2, epsilon);

MSECost MSE(&L2);

NeuralNetwork nn(&crossEntropy, &adam, &L2, learning_rate);
```


#### Constructing the Neural Network
The process of building a neural network involves defining its architecture by sequentially adding layers. Each layer type serves a specific purpose, such as transforming inputs, introducing non-linearity, or producing the final output. Here's a step-by-step breakdown of how to construct the neural network architecture using the example code:

```cpp
nn.addLayer(new LinearLayer("linear1", Dimensions(784, 128)));
nn.addLayer(new ReLUActivation("relu"));
nn.addLayer(new LinearLayer("linear2", Dimensions(128, 64)));
nn.addLayer(new ReLUActivation("relu2"));
nn.addLayer(new LinearLayer("linear3", Dimensions(64, 10)));
nn.addLayer(new SoftmaxActivation("softmax"));
```

Construct the neural network by starting with 784 input nodes, transforming them through hidden layers with Linear and ReLU activations to 128 and 64 nodes, and finally producing class probabilities using a Softmax activation on 10 output nodes for MNIST classification.

#### Training the Neural Network
Train the neural network using the provided architecture and dataset, iterating over epochs and batches. During each iteration, perform forward and backward passes to update the network's weights, while monitoring the cost.

```cpp
Matrix Y;

for (int epoch = 0; epoch < 11; epoch++) 
{
  float cost = 0.0;

  for (int batch = 0; batch < traindata.getNumOfBatches(); batch++) 
  {
    Y = nn.forward(traindata.getBatches().at(batch));
      nn.backprop(Y, traindata.getTargets().at(batch));
        
      LinearLayer* linearLayer = dynamic_cast<LinearLayer*>(nn.getLayers()[2]);
      Matrix layerW = linearLayer->getWeightsMatrix();
      cost += crossEntropy.cost(Y, traindata.getTargets().at(batch), layerW);
  }

  if (epoch % 1 == 0) 
    std::cout << "Epoch: " << epoch << ", Cost: " << cost / traindata.getNumOfBatches() << std::endl;
}
```

#### Evaluating on Test Data
Evaluate the trained neural network on the test data:

```cpp
MNIST testdata(32, 312, "datasets/mnist_test.csv");
Matrix A;

float accuracy = 0.0f;
for (int i = 0; i < testdata.getNumOfBatches(); i++)
{
    A = nn.forward(testdata.getBatches().at(i));
    accuracy += nn.computeAccuracy(A, testdata.getTargets().at(i));
}

std::cout << "Accuracy on test data: " << accuracy / testdata.getNumOfBatches();
```

This code computes the accuracy of your trained model on the test data.

By following the usage instructions provided, you can train and evaluate a neural network using the `CUDA Neural Network Framework`. Customize the architecture and parameters to suit your specific problem and achieve accurate results.

## Benchmark
To assess the performance of the `CUDA Neural Network Framework`, a benchmark comparison was conducted. The neural network architecture created using this framework was replicated in `PyTorch`, with the same layers, activations, cost functions, optimizers, and hyperparameters. The network was trained on the `MNIST dataset` using identical settings in both frameworks.

The results of the benchmark revealed that the network trained using the `CUDA Neural Network Framework` achieved an accuracy of `96%` on the test dataset. Remarkably, when the same architecture and training process were applied to the PyTorch implementation, an identical accuracy of `96%` was obtained on the test data. This benchmark underscores the capability and accuracy of the `CUDA Neural Network Framework`, aligning its performance with established deep learning libraries.

## Contributing
Feel free to download and experiment with the `CUDA Neural Network Framework` for your personal use. However, please note that this project is an ongoing work, and I'll be actively developing it further. As a result, I won't be accepting pull requests for general changes or enhancements at this time.

If you encounter any issues while using the framework or have suggestions for improvement, please don't hesitate to open an issue in the repository. Your feedback is valuable and will contribute to the project's growth. Thank you for your interest in the `CUDA Neural Network Framework`.

