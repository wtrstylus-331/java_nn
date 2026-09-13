package org.sdws.nn;

import org.sdws.util.ActivationFunc;
import org.sdws.mathematics.Activation;
import org.sdws.mathematics.NArray;
import org.sdws.mathematics.NNArray;

public class NetworkLayer {
    public NArray<Number> weights;
    public NArray<Number> biases;
    protected ActivationFunc activation;
    protected int numInputs, numOutputs;
    protected NArray<Number> gradientsW, gradientsB;
    public NArray<Number> preActivationOutput;
    public NArray<Number> postActivationOutput;

    /**
     * Defines a single layer of neurons, initialized with a random set of weights
     * and biases. Activation function type is set to {@link ActivationFunc}{@code .ReLU} by default.<p></p>
     * The initialized layer is represented as a 2-dimensional {@code NNArray}
     * with the shape [<code>neurons</code> x <code>inputs</code>].
     * @param inputs integer representing the amount of inputs for this layer
     * @param neurons integer representing the total amount of neurons for this layer
     */
    public NetworkLayer(int inputs, int neurons) {
        if (neurons < 1) {
            throw new IllegalArgumentException("Neurons in a layer must be greater than or equal to one");
        }

        this.weights = NArray.CreateRandom(neurons, inputs);
        this.biases = NArray.CreateZeros(neurons);
        this.activation = ActivationFunc.ReLU;
        this.numInputs = inputs;
        this.numOutputs = neurons;
        this.gradientsW = NArray.CreateZeros(neurons, inputs);
        this.gradientsB = NArray.CreateZeros(neurons);
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
    public NetworkLayer(int inputs, int neurons, ActivationFunc activation) {
        if (neurons < 1) {
            throw new IllegalArgumentException("Neurons in a layer must be greater than or equal to one");
        }

        this.weights = NArray.InitializeHe(neurons, inputs);
        this.biases = NArray.CreateZeros(neurons);
        this.activation = activation;
        this.numInputs = inputs;
        this.numOutputs = neurons;
        this.gradientsW = NArray.CreateZeros(neurons, inputs);
        this.gradientsB = NArray.CreateZeros(neurons);
    }

    /**
     * Calculate the output(s) of this layer via matrix/vector computation
     * between neuron weights and biases, finally applying the output into
     * the specified activation function initialized for this layer.
     * @param input a {@link NNArray} representing either a matrix or vector,
     *              or simple a {@code Float} in which case its shape and dimension is {@code 1}
     */
    public void ForwardPass(NArray input) {
        this.calculateOutput(input);
        System.out.println("\nOUTPUT (pre): " + this.preActivationOutput+"\n");
        this.postActivationOutput = NArray.DeepCopy(this.preActivationOutput);

        if (this.activation.equals(ActivationFunc.None)) {
            System.out.println("\nOUTPUT: " + this.postActivationOutput +"\n");
            return;
        }

        switch (this.activation) {
            case ReLU -> Activation.ReLU(this.postActivationOutput);
            case Sigmoid -> Activation.Sigmoid(this.postActivationOutput);
            case Step -> Activation.Step(this.postActivationOutput);
            case Tanh -> Activation.Tanh(this.postActivationOutput);
            case Softmax -> Activation.Softmax(this.postActivationOutput);
        }

        System.out.println("\nOUTPUT (post): " + this.postActivationOutput+"\n");
    }

    private void calculateOutput(NArray input) {
        this.preActivationOutput = NArray.Mul(input, this.weights);
        this.preActivationOutput = NArray.Add(this.preActivationOutput, this.biases);
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
