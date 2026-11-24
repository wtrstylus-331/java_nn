package org.sdws.nn;

import org.sdws.mathematics.*;

public class NetworkLayer {
    protected NArray weights;
    protected NArray biases;
    protected Activation activation;
    protected int numInputs, numOutputs;
    protected NArray preActivationOutput;
    protected NArray postActivationOutput;

    protected NetworkLayer prevLayer, nextLayer;

    protected NArray neuronDeltas, weightGrads, biasGrads;

    protected int neurons;

    /**
     * Defines a single layer of neurons, initialized with a random set of weights
     * and biases. Activation function is set to {@link ReLU} by default.<p></p>
     * The initialized layer is represented as a 2-dimensional {@code NNArray}
     * with the shape [<code>neurons</code> x <code>inputs</code>].
     * @param inputs integer representing the amount of inputs for this layer
     * @param neurons integer representing the total amount of neurons for this layer
     */
    public NetworkLayer(int inputs, int neurons) {
        if (neurons < 1) {
            throw new IllegalArgumentException("Neurons in a layer must be greater than or equal to one");
        }

        this.weights = NArray.InitializeHe(neurons, inputs);
        this.biases = NArray.CreateZeros(neurons);
        this.activation = new ReLU();
        this.numInputs = inputs;
        this.numOutputs = neurons;
        this.prevLayer = null;
        this.nextLayer = null;

        this.neuronDeltas = null;
        this.weightGrads = NArray.CreateZeros(neurons, inputs);
        this.biasGrads = NArray.CreateZeros(neurons);

        this.neurons = neurons;
    }

    /**
     * Defines a single layer of neurons, initialized with a random set of weights
     * and biases, and a specified activation function.<p></p>
     * The initialized layer is represented as a 2-dimensional {@code NNArray}
     * with the shape [<code>neurons</code> x <code>inputs</code>].
     * @param inputs integer representing the amount of inputs for this layer
     * @param neurons integer representing the total amount of neurons for this layer
     * @param activation function type implementing the {@link Activation} interface
     */
    public NetworkLayer(int inputs, int neurons, Activation activation) {
        if (neurons < 1) {
            throw new IllegalArgumentException("Neurons in a layer must be greater than or equal to one");
        }

        this.weights = NArray.InitializeHe(neurons, inputs);
        this.biases = NArray.CreateZeros(neurons);
        this.activation = activation;
        this.numInputs = inputs;
        this.numOutputs = neurons;
        this.prevLayer = null;
        this.nextLayer = null;

        this.neuronDeltas = null;
        this.weightGrads = NArray.CreateZeros(neurons, inputs);
        this.biasGrads = NArray.CreateZeros(neurons);

        this.neurons = neurons;
    }

    public int getNeurons() {
        return this.neurons;
    }

    public NetworkLayer getPrevLayer() {
        return this.prevLayer;
    }

    public NetworkLayer getNextLayer() {
        return this.nextLayer;
    }

    public NArray getWeights() {
        return this.weights;
    }

    public void setWeights(NArray weights) {
        this.weights = weights;
    }

    /**
     * Calculate the output(s) of this layer via matrix/vector computation
     * between neuron weights and biases, finally applying the output into
     * the specified activation function initialized for this layer.
     * @param input a {@link NArray} representing either a matrix or vector,
     *              or simple a {@code Float} in which case its shape and dimension is {@code 1}
     */
    protected void forwardPass(NArray input) {
        this.calculateOutput(input);
        this.postActivationOutput = this.activation.compute(this.preActivationOutput);
    }

    protected void forwardPassBatch(NArray input) {
        float[][] array = input.getMatrixArray();
        float[][] preActArray = new float[array.length][this.neurons];
        for (int i = 0; i < array.length; i++) {
            NArray row = NArray.FromElements(array[i]);
            preActArray[i] = NArray.Add(NArray.Mul(this.weights, row), this.biases).getVecArray();
        }
        this.preActivationOutput = NArray.FromNested(preActArray);
        this.postActivationOutput = this.activation.compute(this.preActivationOutput);
    }

    private void calculateOutput(NArray input) {
        this.preActivationOutput = NArray.Mul(this.weights, input);
        this.preActivationOutput = NArray.Add(this.preActivationOutput, this.biases);
    }

    @Override
    public String toString() {
        return "==========================\n" +
                "Weights:\n" +
                this.weights + "\n" +
                "Weight grads:\n" +
                this.weightGrads + "\n" +
                "Biases:\n" +
                this.biases + "\n" +
                "Bias grads:\n" +
                this.biasGrads + "\n" +
                "Neuron deltas:\n" +
                this.neuronDeltas + "\n" +
                "Output:\n" +
                this.postActivationOutput + "\n" +
                "==========================";
    }
}
