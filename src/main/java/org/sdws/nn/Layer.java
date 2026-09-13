package org.sdws.nn;

import org.sdws.util.ActivationFunc;
import org.sdws.mathematics.NNArray;

public class Layer {
    protected NNArray weights;
    protected NNArray biases;
    protected ActivationFunc activation;
    protected int numInputs, numOutputs;
    protected NNArray gradientsW, gradientsB;
    public NNArray preActivationOutput;
    public NNArray postActivationOutput;

    /**
     * Defines a single layer of neurons, initialized with a random set of weights
     * and biases. Activation function type is set to {@link ActivationFunc}{@code .ReLU} by default.<p></p>
     * The initialized layer is represented as a 2-dimensional {@code NNArray}
     * with the shape [<code>neurons</code> x <code>inputs</code>].
     * @param inputs integer representing the amount of inputs for this layer
     * @param neurons integer representing the total amount of neurons for this layer
     */
    public Layer(int inputs, int neurons) {
        if (neurons < 1) {
            throw new IllegalArgumentException("Neurons in a layer must be greater than or equal to one");
        }

        this.weights = NNArray.createRandom(neurons, inputs);
        this.biases = NNArray.createZeros(neurons);
        this.activation = ActivationFunc.ReLU;
        this.numInputs = inputs;
        this.numOutputs = neurons;
        this.gradientsW = NNArray.createZeros(neurons, inputs);
        this.gradientsB = NNArray.createZeros(neurons);
    }

    /**
     * Defines a single layer of neurons, initialized with a random set of weights
     * and biases, and a specified activation function.<p></p>
     * The initialized layer is represented as a 2-dimensional {@code NNArray}
     * with the shape [<code>neurons</code> x <code>inputs</code>].
     * @param inputs integer representing the amount of inputs for this layer
     * @param neurons integer representing the total amount of neurons for this layer
     * @param activation function type from the {@link ActivationFunc} enumerator
     */
    public Layer(int inputs, int neurons, ActivationFunc activation) {
        if (neurons < 1) {
            throw new IllegalArgumentException("Neurons in a layer must be greater than or equal to one");
        }

        this.weights = NNArray.HeInitialization(neurons, inputs);
        this.biases = NNArray.createZeros(neurons);
        this.activation = activation;
        this.numInputs = inputs;
        this.numOutputs = neurons;
        this.gradientsW = NNArray.createZeros(neurons, inputs);
        this.gradientsB = NNArray.createZeros(neurons);
    }

    /**
     * Calculate the output(s) of this layer via matrix/vector computation
     * between neuron weights and biases, finally applying the output into
     * the specified activation function initialized for this layer.
     * @param input a {@link NNArray} representing either a matrix or vector,
     *              or simple a {@code Float} in which case its shape and dimension is {@code 1}
     */
    public void ForwardPass(NNArray input) {
        this.calculateOutput(input);
        System.out.println("\nOUTPUT (pre): " + this.preActivationOutput+"\n");
        this.postActivationOutput = this.preActivationOutput.deepCopy();

        if (this.activation.equals(ActivationFunc.None)) {
            System.out.println("\nOUTPUT: " + this.postActivationOutput +"\n");
            return;
        }

//        switch (this.activation) {
//            case ReLU -> Activation.ReLU(this.postActivationOutput);
//            case Sigmoid -> Activation.Sigmoid(this.postActivationOutput);
//            case Step -> Activation.Step(this.postActivationOutput);
//            case Tanh -> Activation.Tanh(this.postActivationOutput);
//            case Softmax -> Activation.Softmax(this.postActivationOutput);
//        }

        System.out.println("\nOUTPUT (post): " + this.postActivationOutput+"\n");
    }

    private void calculateOutput(NNArray input) {
//        if (input.dimension != 1 || input.shape.get(0) != this.numInputs) {
//            throw new IllegalArgumentException("Input vector length does not match layer input size.");
//        }
//        System.out.println(this.numInputs + " " + this.numOutputs);
//
//        if (input.dimension != 1) {
//            throw new IllegalArgumentException("clause 1");
//        }
//
//        if (input.shape.get(0) != this.numInputs) {
//            throw new IllegalArgumentException("clause 2");
//        }

        this.preActivationOutput = NNArray.dot(input, this.weights);
        this.preActivationOutput = NNArray.add(this.preActivationOutput, this.biases);
    }

    @Override
    public String toString() {
        return "==========================\n" +
                "Weights:\n" +
                this.weights + "\n" +
                "Biases:\n" +
                this.biases + "\n" +
                "==========================";
    }
}
