# SkyNeuralNet
A program written in C++ and CUDA for creating, training and exporting fully connected feedforward neural networks. It can be easily configured to mainly use the CPU or GPU for computations, depending on the size of the network (e.g. CPU for very small networks and GPU for bigger networks).

The program can load and interpret either arbitrary files or images (using the stb_image library) to use as training data.

# Network structure
Each layer contains a predefined number of neurons, with each layer also containing one extra bias neuron. While not necessary, the output layer also contains a bias neuron to generalize the algorithms used. Neurons in the hidden layers use the ReLU activation function, while neurons in the output layer use a sigmoid function.

# How the GPU is utilized
To keep the program simple when utilizing shared memory on the GPU, a maximum of 1024 neurons in a single layer is allowed. Anything beyond that will automatically force the program to only use the CPU.

### Forward propagation: 
The output of each neuron in a single layer is computed in parallel, with each layer being evaluated sequentially. Shared memory is used here to load in the previous layer's output values. 

The majority of this process is computed on the GPU. However, the activation function for the output layer is calculated on the CPU to avoid precision errors when computing the sigmoid function. The loss when using the CPU like this is extremely minimal and the increased precision is preferred.

### Backward propagation: 
The gradient of each neuron in the output layer is computed on the CPU, again, to keep precision when computing derivatives of the sigmoid function. After that, the GPU is used to compute gradients in the hidden layers, similar to how forward propagation is executed. Shared memory is utilized here aswell to load in the next layer's gradients. 

Updating the weights also takes place on the GPU, since each weight can be computed independently of eachother.

# Example Usage
An example application of using a neural network exported by SkyNeuralNet can be found here:
[https://github.com/SiTronXD/SkyNeuralNetUsageExample](https://github.com/SiTronXD/SkyNeuralNetUsageExample)
