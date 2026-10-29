~ An attempt at making a Java library for creating, evaluating, and utilizing Neural Networks, with total flexibility in mind. This project originally started with a desire to learn and implement Neural Networks for areas such as Reinforcement Learning (which will hopefully be implemented via Java GUI environments/games). The reason I started with Java, rather than Python, was to get a sense of how a complex system such as an ANN would work in an Object-Oriented, statically typed language, which would also make it simple to implement the same idea in other languages like C#.

# Overview
### Implemented
- Mathematical logic for vectors and matrices, collapsed into a single class **_NArray_**, to dynamically handle both cases for flexibility and efficiency as a result of inspiration from the Python _numpy_ library
- Various activation functions and algorithms (Step, Sigmoid, ReLU, Softmax), along with feed forward algorithm for the _**NeuralNetwork**_ class
- Various methods of calculating loss divided into cases for single vector input/output and batches of multiple vectors (as matrices); loss algorithms being Mean Squared Error, Root MSE, Cross-Categorical Entropy, Binary Categorical Entropy
- GUI class solely meant to display Neural Networks
- Backpropagation algorithms for CCE + Softmax and MSE (not handling Softmax) to handle both single vector inputs (stochastic gradient descent) and batches of vector inputs as matrices (batch gradient descent)

### TODO
- Implement gradient descent algorithms for remaining loss functions for the backpropagation process
- Optimizers for gradient descent
- Implement MVC design principles to structure the core GUI application with features such as manual forward/backward passing and value inspection for individual neurons
- As usual, proper documentation for methods/functions

### Use cases
- The **_Main_** class is currently being used as a "sandbox" to test out functionality of the library to ensure each component is working as intended
- Playing around with it visually by calling the _run()_ method from the **_NeuralVisualizer_** class
- Potentially for Reinforcement Learning regarding Java environments

The current architecture of this library is constantly changing to accommodate for future ideas/necessities and to ensure that programs made with this architecture will compile and run as desired, with no errors.
#### Architecture:
- **_NArray_** is the main class that holds logic for vector/matrix operations for floats only
- **_Activation_** is the class that holds logic for computing values into their respective activation function
- **_NetworkLayer_** is the class that enables the user to create neurons and specify an activation function for all neurons in this layer, representing weights and biases via the **_NArray_** class
- **_NeuralNetwork_** is the class where you can assemble layer objects into a network, with a dedicated _feedForward_ method and attributes to access loss and output for testing purposes
- **_ActivationFunc_** & **_LossAlgorithm_** are enums containing the currently implemented activation functions and loss calculation algorithms, which are mentioned and used in various methods for the previous classes
- **_NeuralVisualizer_** is a JavaFX application class meant to display components of a Neural Network created by the user to a hand-crafted GUI, the goal is to be able to manually feed forward/backward between layers and inspect weights/biases, with more features to come

# Playing around with it
1. Clone via git or GitHub desktop
2. Check out the example usage below to see how it works, and editing the **_Main_** class

## Example usage
First we create three layers.<br>
It is important that the # of neurons of the **i-1**th layer is equivalent to the input argument for the **i**th layer (except for the first layer)<br>
Also note that it is not strictly necessary to instantiate a totally seperate layer for the softmax calculation, as this setup could also work with only 2 layers (one ReLU and the last one being Softmax) having six and four neurons respectively, though the extra softmax layer is shown for demonstration and clarity purposes.
```
NetworkLayer first = new NetworkLayer(3,6, ActivationFunc.ReLU); // this layer takes in 3 inputs and has 6 neurons (outputs)
NetworkLayer second = new NetworkLayer(6,4, ActivationFunc.ReLU); // this layer takes in 6 inputs and has 4 neurons (outputs)
NetworkLayer softmax = new NetworkLayer(4,4, ActivationFunc.Softmax); // this layer takes in 4 inputs and has 4 neurons (outputs)
```
<br>

Now we initialize a new NeuralNetwork object.<br>
Note that the third parameter for the NeuralNetwork class is a varargs parameter, thus you can add as many layers as possible, but it would be advisable to keep it non-empty.
```
NeuralNetwork network = new NeuralNetwork(
        0.001f, // learning rate
        LossAlgorithm.CCELoss, // loss algorithm via enum
        first, second, softmax // layer arguments
);
```
<br>

Now we create an input vector of size 3 for the 3 inputs of the first layer using one of the static functions of the NArray class. This function specifically initializes and assigns our variable a representation of a vector with 3 randomly generated float values.
```
NArray testInput = NArray.CreateRandom(3);
```
<br>

And we also create the desired output of size 2 since there are 2 neurons in the output layer.<br>
Note that in this case, since our loss algorithm is Cross Categorical Entropy Loss (CCE), and that the last layer's activation function utilizes the Softmax algorithm, it is recommended that our desired output is in the form of a one-hot encoded vector. In this case, our desired output is the 2nd class (1st index).
```
NArray testOutput = NArray.FromElements(0f, 0f, 0f, 1f);
```
<br>

Now we set the input for our neural network as our vector input created earlier and call the feedForward method, which passes our input attribute through each layer and calculates the loss obtained by the neural network.
```
network.input(testInput);
network.feedForward();
network.calculateLoss(testOutput);
```
<br>

And here is what the sample print statement would look like:
```
System.out.println("loss: " + network.loss() + " output: " + network.output);
>>> loss: 1.6837437 output: [0.16583891, 0.34242353, 0.30606005, 0.18567756]
```
In this case, the loss is quite high since our one-hot value from our desired output differs significantly (the 3rd indices), relative to the values of the other indices.
<br><br>
(note that the weights are always randomly generated when creating new layers, so every time you run the program the loss and output will be completely random, until backpropagation and descent opimization is implemented that is)
<br><br>
### Contributions regarding the implementation of various remaining aspects of this library are always welcome.
