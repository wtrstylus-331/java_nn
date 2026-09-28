~ An attempt at making a Java library for creating, evaluating, and utilizing Neural Networks, with total flexibility in mind. This project originally started with a desire to learn and implement Neural Networks for areas such as Reinforcement Learning (which will hopefully be implemented via Java GUI environments/games). The reason I started with Java, rather than Python, was to get a sense of how a complex system such as an ANN would work in an Object-Oriented, statically typed language, which would also make it easier to implement the same idea in other languages like C#.

# Overview
### Implemented
- Mathematical logic for vectors and matrices, collapsed into a single class **_NArray_**, to dynamically handle both cases for flexibility and efficiency as a result of inspiration from the Python _numpy_ library
- Various activation functions and algorithms (Step, Sigmoid, ReLU, Softmax), along with feed forward algorithm for the _**NeuralNetwork**_ class
- Various methods of calculating loss divided into cases for single vector input/output and batches of multiple vectors (as matrices); loss algorithms being Mean Squared Error, Root MSE, Cross-Categorical Entropy, Binary Categorical Entropy

### TODO
- Implement gradient descent algorithms for various loss functions for the backpropagation process
- Optimizers for gradient descent
- Loops for training NN's upon implementation of the previous bullet points
- As usual, proper documentation for methods/functions

### Use cases
- The **_Main_** class is currently being used as a "sandbox" to test out functionality of the library to ensure each component is working as intended
- Potentially for Reinforcement Learning regarding Java environments

The current architecture of this library is constantly changing to accommodate for future ideas/necessities and to ensure that programs made with this architecture will compile and run as desired, with no errors.
#### Architecture:
- **_NArray_** is the main class that holds logic for vector/matrix operations (**_NNArray_** is a deprecated version as the former is meant to be more flexible in accepting and manipulating Number data types by treating it as a generic type within the class)
- **_Activation_** is the class that holds logic for computing values into their respective activation function
- **_NetworkLayer_** is the class that enables the user to create neurons and specify an activation function for all neurons in this layer, representing weights and biases via the **_NArray_** class (**_Layer_** is a deprecated version for a similar reason mentioned above with **_NArray_**, simply refactoring logic to work with Number types via generic types)
- **_NeuralNetwork_** is the class where you can assemble layer objects into a network, with a dedicated _feedForward_ method and attributes to access loss and output for testing purposes
- **_ActivationFunc_** & **_LossAlgorithm_** are enums containing the currently implemented activation functions and loss calculation algorithms, which are mentioned and used in various methods for the previous classes

# Playing around with it
git clone https://github.com/wtrstylus-331/java_nn.git
cd java_nn
