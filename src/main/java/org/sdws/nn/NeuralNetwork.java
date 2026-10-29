package org.sdws.nn;

import org.sdws.mathematics.Activation;
import org.sdws.util.ActivationFunc;
import org.sdws.util.LossAlgorithm;
import org.sdws.mathematics.NArray;

import java.util.Arrays;
import java.util.Objects;

public class NeuralNetwork {
    private NetworkLayer[] layers;
    private LossAlgorithm lossAlgorithm;

    private NArray input, desiredOutput;
    public Object output;

    private float loss;
    private float learningRate;

    public NeuralNetwork(NetworkLayer... layers) {
        this.layers = new NetworkLayer[layers.length];
        this.learningRate = 0.0001f;
        this.loss = Float.NaN;
        this.lossAlgorithm = LossAlgorithm.CCELoss;
        this.output = null;

        if (layers.length == 1) {
            this.layers[0] = layers[0];
        } else if (layers.length == 2) {
            this.layers[0] = layers[0];
            this.layers[1] = layers[1];

            this.layers[0].nextLayer = this.layers[1];
            this.layers[1].prevLayer = this.layers[0];
        } else {
            for (int i = 0; i < layers.length; i++) {
                NetworkLayer layer = layers[i];

                if (i == 0) {
                    layer.nextLayer = layers[i + 1];
                } else if (i == layers.length - 1) {
                    layer.prevLayer = layers[i - 1];
                } else {
                    layer.prevLayer = layers[i - 1];
                    layer.nextLayer = layers[i + 1];
                }

                this.layers[i] = layer;
            }
        }
    }

    public NeuralNetwork(float learningRate, LossAlgorithm lossAlg, NetworkLayer... layers) {
        this.layers = new NetworkLayer[layers.length];
        this.learningRate = learningRate;
        this.loss = Float.NaN;
        this.lossAlgorithm = lossAlg;
        this.output = null;

        if (layers.length == 1) {
            this.layers[0] = layers[0];
        } else if (layers.length == 2) {
            this.layers[0] = layers[0];
            this.layers[1] = layers[1];

            this.layers[0].nextLayer = this.layers[1];
            this.layers[1].prevLayer = this.layers[0];
        } else {
            for (int i = 0; i < layers.length; i++) {
                NetworkLayer layer = layers[i];

                if (i == 0) {
                    layer.nextLayer = layers[i + 1];
                } else if (i == layers.length - 1) {
                    layer.prevLayer = layers[i - 1];
                } else {
                    layer.prevLayer = layers[i - 1];
                    layer.nextLayer = layers[i + 1];
                }

                this.layers[i] = layer;
            }
        }
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float learningRate() {
        return this.learningRate;
    }

    public void setLoss(LossAlgorithm lossAlgorithm) {
        this.lossAlgorithm = lossAlgorithm;
    }

    public LossAlgorithm lossAlgorithm() {
        return this.lossAlgorithm;
    }

    public float loss() {
        return this.loss;
    }

    public NetworkLayer[] getLayers() {
        return this.layers;
    }

    public void input(NArray input) {
        this.input = input;
    }

    public NArray getInput() {
        return this.input;
    }

    public void setDesiredOutput(NArray desiredOutput) {
        this.desiredOutput = desiredOutput;
    }

    public NArray getDesiredOutput() {
        return this.desiredOutput;
    }

    /**
     * for debug purposes
     */
    public void printLayers() {
        for (int i = 0; i < this.layers.length; i++) {
            System.out.println(this.layers[i].toString());
        }
    }

    public void erase() {
        this.layers = null;
        this.output = null;
        this.loss = Float.NaN;
        this.learningRate = 0.0001f;
        this.lossAlgorithm = null;
    }

    public void forwardPass() {
        if (this.input == null) return;

        switch (this.input.dimension) {
            default: {  // single vector input
                NArray prevOutput = this.input;
                for (NetworkLayer layer : this.layers) {
                    layer.forwardPass(prevOutput);
                    prevOutput = layer.postActivationOutput;
                }

                this.output = prevOutput;
                break;
            }
            case 2: {   // batch of vectors (matrix) as input
//                float[][] batch = this.input.getMatrixArray();
//                float[][] batchOutput = new float[batch.length][this.layers[this.layers.length - 1].getNeurons()];
//
//                for (int i = 0; i < batch.length; i++) {
//                    NArray prevOutput = NArray.FromRows(batch[i]);
//                    for (NetworkLayer layer : this.layers) {
//                        layer.forwardPass(prevOutput);
//                        prevOutput = layer.postActivationOutput;
//                    }
//
//                    batchOutput[i] = prevOutput.getVecArray();
//                }
//
//                this.output = NArray.FromNested(batchOutput);
//                break;

                NArray prevOutput = this.input;

                for (NetworkLayer layer : this.layers) {
                    layer.forwardPassBatch(prevOutput);
                    prevOutput = layer.postActivationOutput;
                }

                this.output = prevOutput;
                break;
            }
        }
    }

    public void backwardPass(NArray desiredOutput) {
        if (this.input == null || this.output == null) {
            return;
        }

        switch (this.lossAlgorithm) {
            case CCELoss: {
                if (this.layers[this.layers.length - 1].activation == ActivationFunc.Softmax) {
                    this.backpropCCESoftmax(desiredOutput);
                    this.updateParams();
                }
                break;
            }
            case MSELoss: {
                if (this.layers[this.layers.length - 1].activation != ActivationFunc.Softmax) {
                    this.backpropMSE(desiredOutput);
                    this.updateParams();
                } else {
                    throw new UnsupportedOperationException("The configuration of the loss algorithm and neural network provided for backpropagation has not been implemented.");
                }
                break;
            }
            default: {
                throw new UnsupportedOperationException("The configuration of the loss algorithm and neural network provided for backpropagation has not been implemented.");
            }
        }
    }

    private void backpropMSE(NArray desiredOutput) {
        if (desiredOutput.dimension == 2) {
            this.MSEBatchBackprop(desiredOutput);
        } else {
            this.MSESingleVectorBackprop(desiredOutput);
        }
    }

    private void MSEBatchBackprop(NArray desiredOutput) {
        int batchSize = desiredOutput.shape[0];

        for (int b = 0; b < batchSize; b++) {
            for (int i = this.layers.length - 1; i >= 0; i--) {
                NetworkLayer current = layers[i];
                int neurons = current.getNeurons();

                if (current.neuronDeltas == null) {
                    current.neuronDeltas = NArray.CreateZeros(batchSize, neurons);
                }

                if (i == this.layers.length - 1) {  // output layer
                    for (int j = 0; j < neurons; j++) {
                        float val = 2 * (current.postActivationOutput.get(j) - desiredOutput.get(j));
                        if (val == -0.0f) {
                            val = 0.0f;
                        }
                        current.neuronDeltas.set(b, j, val);
                    }
                } else {    // all other layers
                    NetworkLayer next = current.nextLayer;
                    float[][] nextWeights = next.weights.getMatrixArray();
                    int nextNeurons = next.neurons;

                    for (int j = 0; j < neurons; j++) {
                        float sum = 0f;
                        for (int k = 0; k < nextNeurons; k++) {
                            sum += next.neuronDeltas.get(b, k) * nextWeights[k][j];
                        }

                        float val = 0f;
                        switch (current.activation) {
                            case ReLU -> val = sum * Activation.derivativeReLU(current.postActivationOutput.get(b, j));
                            case Sigmoid -> val = sum * Activation.derivativeSigmoid(current.postActivationOutput.get(b, j));
                        }

                        if (val == -0.0f) {
                            val = 0.0f;
                        }
                        current.neuronDeltas.set(b, j, val);
                    }
                }
            }
        }

        for (int i = this.layers.length - 1; i >= 0; i--) {
            NetworkLayer current = layers[i];
            float[][] weights = current.weights.getMatrixArray();
            int neurons = current.getNeurons();

            for (int j = 0; j < neurons; j++) {
                float addedDelta = 0f;
                for (int b = 0; b < batchSize; b++) {
                    addedDelta += current.neuronDeltas.get(b, j);
                }

                float biasGrad = addedDelta / batchSize;
                current.biasGrads.set(j, biasGrad == -0.0f ? 0f : biasGrad);

                for (int b = 0; b < batchSize; b++) {
                    for (int k = 0; k < weights[j].length; k++) {
                        float weightGrad = addedDelta * (i == 0 ? this.input.get(b,k) : current.prevLayer.postActivationOutput.get(b, k));
                        current.weightGrads.set(j, k, weightGrad == -0.0f ? 0f : weightGrad);
                    }
                }
            }
        }
    }

    private void MSESingleVectorBackprop(NArray desiredOutput) {
        for (int i = this.layers.length - 1; i >= 0; i--) {
            NetworkLayer current = layers[i];
            float[][] weights = current.weights.getMatrixArray();
            int neurons = current.getNeurons();

            if (current.neuronDeltas == null) {
                current.neuronDeltas = NArray.CreateZeros(neurons);
            }

            NArray previous = (i == 0) ? this.input : current.prevLayer.postActivationOutput;

            if (i == this.layers.length - 1) {  // output layer
                for (int j = 0; j < neurons; j++) {
                    float val = 2 * (current.postActivationOutput.get(j) - desiredOutput.get(j));
                    if (val == -0.0f) {
                        val = 0.0f;
                    }
                    current.neuronDeltas.set(j, val);
                }
            } else {    // all other layers
                this.calculateLayerDeltas(current, neurons);
            }

            // assign gradients of weights/biases
            this.assignGrads(current, neurons, weights, previous);
        }
    }

    private void backpropCCESoftmax(NArray desiredOutput) {
        if (desiredOutput.dimension == 2) {
            this.CCESoftmaxBatchBackprop(desiredOutput);
        } else {
            this.CCESoftmaxSingleVectorBackprop(desiredOutput);
        }
    }

    private void CCESoftmaxBatchBackprop(NArray desiredOutput) {
        int batchSize = desiredOutput.shape[0];

        for (int b = 0; b < batchSize; b++) {
            for (int i = this.layers.length - 1; i >= 0; i--) {
                NetworkLayer current = layers[i];
                int neurons = current.getNeurons();

                if (current.neuronDeltas == null) {
                    current.neuronDeltas = NArray.CreateZeros(batchSize, neurons);
                }

                if (i == this.layers.length - 1) {  // output layer
                    for (int j = 0; j < neurons; j++) {
                        float val = current.postActivationOutput.get(b, j) - desiredOutput.get(b, j);
                        if (val == -0.0f) {
                            val = 0.0f;
                        }
                        current.neuronDeltas.set(b, j, val);
                    }
                } else {    // all other layers
                    NetworkLayer next = current.nextLayer;
                    float[][] nextWeights = next.weights.getMatrixArray();
                    int nextNeurons = next.neurons;

                    for (int j = 0; j < neurons; j++) {
                        float sum = 0f;
                        for (int k = 0; k < nextNeurons; k++) {
                            sum += next.neuronDeltas.get(b, k) * nextWeights[k][j];
                        }

                        float val = 0f;
                        switch (current.activation) {
                            case ReLU -> val = sum * Activation.derivativeReLU(current.postActivationOutput.get(b, j));
                            case Sigmoid -> val = sum * Activation.derivativeSigmoid(current.postActivationOutput.get(b, j));
                        }

                        if (val == -0.0f) {
                            val = 0.0f;
                        }
                        current.neuronDeltas.set(b, j, val);
                    }
                }
            }
        }

        for (int i = this.layers.length - 1; i >= 0; i--) {
            NetworkLayer current = layers[i];
            float[][] weights = current.weights.getMatrixArray();
            int neurons = current.getNeurons();

            for (int j = 0; j < neurons; j++) {
                float addedDelta = 0f;
                for (int b = 0; b < batchSize; b++) {
                    addedDelta += current.neuronDeltas.get(b, j);
                }

                float biasGrad = addedDelta / batchSize;
                current.biasGrads.set(j, biasGrad == -0.0f ? 0f : biasGrad);

                for (int b = 0; b < batchSize; b++) {
                    for (int k = 0; k < weights[j].length; k++) {
                        float weightGrad = addedDelta * (i == 0 ? this.input.get(b,k) : current.prevLayer.postActivationOutput.get(b, k));
                        current.weightGrads.set(j, k, weightGrad == -0.0f ? 0f : weightGrad);
                    }
                }
            }
        }
    }

    private void CCESoftmaxSingleVectorBackprop(NArray desiredOutput) {
        for (int i = this.layers.length - 1; i >= 0; i--) {
            NetworkLayer current = layers[i];
            float[][] weights = current.weights.getMatrixArray();
            int neurons = current.getNeurons();

            if (current.neuronDeltas == null) {
                current.neuronDeltas = NArray.CreateZeros(neurons);
            }

            NArray previous = (i == 0) ? this.input : current.prevLayer.postActivationOutput;

            if (i == this.layers.length - 1) {  // output layer
                for (int j = 0; j < neurons; j++) {
                    float val = current.postActivationOutput.get(j) - desiredOutput.get(j);
                    if (val == -0.0f) {
                        val = 0.0f;
                    }
                    current.neuronDeltas.set(j, val);
                }
            } else {    // all other layers
                this.calculateLayerDeltas(current, neurons);
            }

            // assign gradients of weights/biases
            this.assignGrads(current, neurons, weights, previous);
        }
    }

    private void calculateLayerDeltas(NetworkLayer currentLayer, int currentNeurons) {
        NetworkLayer next = currentLayer.nextLayer;
        float[][] nextWeights = next.weights.getMatrixArray();
        int nextNeurons = next.neurons;

        for (int j = 0; j < currentNeurons; j++) {
            float sum = 0f;
            for (int k = 0; k < nextNeurons; k++) {
                sum += next.neuronDeltas.get(k) * nextWeights[k][j];
            }

            float val = 0f;
            switch (currentLayer.activation) {
                case ReLU -> val = sum * Activation.derivativeReLU(currentLayer.postActivationOutput.get(j));
                case Sigmoid -> val = sum * Activation.derivativeSigmoid(currentLayer.postActivationOutput.get(j));
            }

            if (val == -0.0f) {
                val = 0.0f;
            }
            currentLayer.neuronDeltas.set(j, val);
        }
    }

    private void assignGrads(NetworkLayer layer, int neurons, float[][] weights, NArray previous) {
        for (int j = 0; j < neurons; j++) {
            float delta = layer.neuronDeltas.get(j);

            layer.biasGrads.set(j, delta);

            for (int k = 0; k < weights[j].length; k++) {
                layer.weightGrads.set(j, k, delta * previous.get(k) == -0.0f ? 0f : delta * previous.get(k));
            }
        }
    }
    
    private void updateParams() {
        for (int i = 0; i < this.layers.length; i++) {
            int neurons = this.layers[i].neurons;

            for (int j = 0; j < neurons; j++) {
                int weightsLen = this.layers[i].weights.getMatrixArray()[j].length;

                for (int k = 0; k < weightsLen; k++) {
                    float currWeight = this.layers[i].weights.get(j, k);
                    float weightGrad = this.layers[i].weightGrads.get(j, k);
                    // grad descent
                    this.layers[i].weights.set(j, k, currWeight - (this.learningRate * weightGrad));
                }

                float currBias = this.layers[i].biases.get(j);
                float biasGrad = this.layers[i].biasGrads.get(j);
                // grad descent
                this.layers[i].biases.set(j, currBias - (this.learningRate * biasGrad));
            }
        }
    }

    /**
     * Calculate the loss of the neural network based on vector outputs from single inputs, or
     * based on matrix outputs as part of batches of vector inputs represented as matrices.
     * @param desiredOutput
     */
    public void calculateLoss(NArray desiredOutput) {
        if (desiredOutput == null || desiredOutput.dimension > 2 || this.lossAlgorithm == null) {
            throw new IllegalArgumentException("Desired output array must not be null to calculate loss.");
        }

        switch (this.lossAlgorithm) {
            case CCELoss -> calculateCCE(desiredOutput);
            case RMSELoss -> calculateRMSE(desiredOutput);
            case BCELoss -> calculateBCE(desiredOutput);
            default -> calculateMSE(desiredOutput);
        }
    }

    /**
     * Intended to calculate loss solely in cases where the neural network produces a single float
     * output (1 neuron in the last layer).
     * @param desiredOutput
     */
    public void calculateLoss(float desiredOutput) {
        if (Float.isNaN(desiredOutput) || this.lossAlgorithm == null) {
            throw new IllegalArgumentException("Desired output array must not be null to calculate loss.");
        }

        switch (this.lossAlgorithm) {
            case CCELoss -> calculateCCE(desiredOutput);
            case RMSELoss -> calculateRMSE(desiredOutput);
            case BCELoss -> calculateBCE(desiredOutput);
            default -> calculateMSE(desiredOutput);
        }
    }

    private void calculateBCE(float desiredOutput) {
        if (!(this.output instanceof Number)) {
            throw new IllegalStateException("Cannot compute BCE loss when neural network output is not of a numeric type.");
        }

        if (desiredOutput == 1f || desiredOutput == 0f) {
            this.loss = desiredOutput * (float)Math.log(((float) this.output)) +
                    (1 - desiredOutput) * (float)Math.log(1 - ((float) this.output));
        } else {
            throw new IllegalArgumentException("Desired output must be either 0.0 or 1.0.");
        }
    }

    private void calculateBCE(NArray desiredOutput) {
        if (!validForBCE(desiredOutput)) {
            return;
        }

        if (desiredOutput.dimension == 1) {
            if (!(((NArray) this.output).dimension == 1)) {
                throw new IllegalArgumentException("Cannot calculate loss from output matrix when desired output is a vector.");
            }

            int n = desiredOutput.getVecArray().length;
            float total = 0f;

            for (int i = 1; i < n; i++) {
                total += desiredOutput.get(i) * (float)Math.log(((NArray)this.output).get(i)) +
                        (1 - desiredOutput.get(i)) * (float)Math.log(1 - ((NArray)this.output).get(i));
            }

            this.loss = (-1f / n) * total;
        } else {
            if (!(((NArray) this.output).dimension == 2)) {
                throw new IllegalArgumentException("Cannot calculate loss from output vector when desired output is a matrix representing batches of outputs.");
            }

            int n = desiredOutput.shape[1];
            float batchesTotal = 0f;

            for (int i = 1; i < desiredOutput.getMatrixArray().length; i++) {
                float batchLoss = 0f;

                for (int j = 1; j < n; j++) {
                    batchLoss += desiredOutput.get(i,j) * (float)Math.log(((NArray)this.output).get(i,j)) +
                            (1 - desiredOutput.get(i,j)) * (float)Math.log(1 - ((NArray)this.output).get(i,j));
                }

                batchesTotal += (-1f / n) * batchLoss;
            }

            this.loss = batchesTotal / ((float) desiredOutput.getMatrixArray().length);
        }
    }

    private void calculateCCE(float desiredOutput) {
        if (!(this.output instanceof Number)) {
            throw new IllegalStateException("Cannot compute CCE loss when neural network output is not of a numeric type.");
        }

        if (desiredOutput == 1f || desiredOutput == 0f) {
            this.loss = -desiredOutput * (float) Math.log((Float)this.output);
        } else {
            throw new IllegalArgumentException("Desired output must be either 0.0 or 1.0.");
        }
    }

    private void calculateCCE(NArray desiredOutput) {
        if (!validForCCE(desiredOutput)) {
            return;
        }

        if (desiredOutput.dimension == 1) {
            if (!Objects.equals(((NArray) this.output).shape[1], desiredOutput.shape[1])) {
                throw new IllegalArgumentException("One-hot vector length for CCE is not equal to network output.");
            }

            int index = -1;
            for (int i = 0; i < desiredOutput.elements; i++) {
                if (desiredOutput.get(i) == 1f) {
                    index = i;
                    break;
                }
            }

            if (index < 0) {
                throw new IllegalArgumentException("No index containing a one-hot value from desired output to calculate CCE loss.");
            }

            this.loss = (float) -Math.log(((NArray)this.output).get(index));
        } else {
            if (!Arrays.equals(((NArray) this.output).shape, desiredOutput.shape)) {
                throw new IllegalArgumentException("One-hot batch for CCE is not equal to network output batch.");
            }

            float sum = 0f;
            int n = (((NArray)this.output).shape[0]);

            float[][] array = desiredOutput.getMatrixArray();
            for (int i = 0; i < n; i++) {
                int index = -1;
                for (int j = 0; j < array[i].length; j++) {
                    if (desiredOutput.get(i, j) == 1f) {
                        index = j;
                        break;
                    }
                }

                float clippedValue = Util.clip(((NArray)this.output).get(i, index));
                sum += (float) Math.log(clippedValue);
            }

            this.loss = ((float) -1/n) * sum;
        }
    }

    private boolean validForCCE(NArray desiredOutput) {
        if (desiredOutput == null) {
            throw new IllegalArgumentException("Desired output for CCE loss must not be null.");
        }

        if (!Util.isOneHot(desiredOutput)) {
            throw new IllegalArgumentException("Vector/matrix desired output must be one-hot encoded for CCE loss.");
        }

        return true;
    }

    private boolean validForBCE(NArray desiredOutput) {
        if (desiredOutput == null) {
            throw new IllegalArgumentException("Desired output for BCE loss must not be null.");
        }

        if (!Objects.equals(this.layers[this.layers.length - 1].activation, ActivationFunc.Sigmoid)) {
            throw new IllegalArgumentException(
                    "The layer being used to calculate BCE Loss must have its activation function set to the Sigmoid activation function (ActivationFunc.Sigmoid)."
            );
        }

        if (!Util.constrainedForBCE(this.layers[this.layers.length - 1].postActivationOutput)) {
            throw new IllegalStateException(
                    "Outputs from Sigmoid activation function layer must be bounded between the open interval of (0,1) for BCE loss calculation."
            );
        }

        if (!Util.isMultiHot(desiredOutput)) {
            throw new IllegalArgumentException("Vector/matrix desired output must be multi-hot encoded with either 1 or 0 as desired output values for BCE loss.");
        }

        return true;
    }

    private void calculateMSE(float desiredOutput) {
        if (!(this.output instanceof Number)) {
            throw new IllegalStateException("Cannot compute MSE loss when neural network output is not of a numeric type.");
        }

        this.loss = (float)Math.pow((Float)this.output - desiredOutput, 2);
    }

    private void calculateMSE(NArray desiredOutput) {
        if (desiredOutput.dimension == 1) { // single output from single input
            if (!validforMSE(desiredOutput)) {
                return;
            }

            float n = desiredOutput.getVecArray().length;
            float sum = 0f;

            for (int i = 0; i < n; i++) {
                sum += (float) Math.pow((((NArray) this.output).get(i) - desiredOutput.get(i)), 2);
            }

            this.loss = sum / n;
        } else { // batch of outputs from batch of inputs
            if (!validforMSE(desiredOutput)) {
                return;
            }

            float totalBatches = desiredOutput.getMatrixArray().length;
            float innerTotal = (desiredOutput.getMatrixArray()[0]).length;
            float sum;
            float batchSum = 0f;

            for (int i = 0; i < totalBatches; i++) {
                sum = 0f;

                for (int j = 0; j < innerTotal; j++) {
                    sum += (float) Math.pow((((NArray) this.output).get(i,j) - desiredOutput.get(i,j)), 2);
                }

                batchSum += sum / innerTotal;
            }

            this.loss = batchSum / totalBatches;
        }
    }

    private void calculateRMSE(float desiredOutput) {
        calculateMSE(desiredOutput);
        this.loss = (float) Math.sqrt(this.loss);
    }

    private void calculateRMSE(NArray desiredOutput) {
        if (desiredOutput.dimension == 1) { // single output from single input
            if (!validforMSE(desiredOutput)) {
                return;
            }

            float n = desiredOutput.getVecArray().length;
            float sum = 0f;

            for (int i = 0; i < n; i++) {
                sum += (float) Math.pow((((NArray) this.output).get(i) - desiredOutput.get(i)), 2);
            }

            this.loss = (float) Math.sqrt(sum / n);
        } else { // batch of outputs from batch of inputs
            if (!validforMSE(desiredOutput)) {
                return;
            }

            float totalBatches = desiredOutput.getMatrixArray().length;
            float innerTotal = (desiredOutput.getMatrixArray()[0]).length;
            float sum;
            float batchSum = 0f;

            for (int i = 0; i < totalBatches; i++) {
                sum = 0f;

                for (int j = 0; j < innerTotal; j++) {
                    sum += (float) Math.pow((((NArray) this.output).get(i,j) - desiredOutput.get(i,j)), 2);
                }

                batchSum += (float) Math.sqrt(sum / innerTotal);
            }

            this.loss = (float) Math.sqrt(batchSum / totalBatches);
        }
    }

    private boolean validforMSE(NArray desiredOutput) {
        if (desiredOutput == null) {
            throw new IllegalArgumentException("Desired output cannot be null for MSE loss.");
        }

        if (desiredOutput.dimension == 1) {
            if (desiredOutput.getVecArray().length == 0) {
                throw new IllegalArgumentException("Desired output vector must not be empty for MSE loss. (single output)");
            }

            if (!(Objects.equals(((NArray)this.output).shape[0], desiredOutput.shape[0]) && Objects.equals(((NArray)this.output).shape[1], desiredOutput.shape[1]))) {
                throw new IllegalArgumentException("Desired output vector must match the length of neural network output for MSE loss. (single output)");
            }
        } else {
            if (desiredOutput.getMatrixArray().length == 0 || (desiredOutput.getMatrixArray()[0]).length == 0) {
                throw new IllegalArgumentException("Desired output matrix must not be empty for MSE loss. (batch output)");
            }

            if (!Objects.equals(((NArray)this.output).shape, desiredOutput.shape)) {
                throw new IllegalArgumentException("Desired output matrix must match the length of neural network output for MSE loss. (batch output)");
            }
        }

        return true;
    }

    private static class Util {
        private static Float clip(float value) {
            if (value < 1e-7f) {
                return 1e-7f;
            }  else if (value > 1f - 1e-7f) {
                return 1f-1e-7f;
            }
            return value;
        }

        /**
         * Returns {@code true} or {@code false} if the input vector
         * represented by the {@link NArray} object is a one-hot vector or not.
         * @param input a {@link NArray} object representing a vector
         * @return a boolean value of either {@code true} or {@code false}
         */
        private static boolean isOneHot(NArray input) {
            int count = 0;
            if (input.dimension == 1) {
                for (int i = 0; i < input.getVecArray().length; i++) {
                    if (input.get(i) == 1f) {
                        count++;
                    }
                }

                return count == 1;
            } else {
                float[][] array = input.getMatrixArray();
                for (int i = 0; i < array.length; i++) {
                    for (int j = 0; j < array[i].length; j++) {
                        if (input.get(i,j) == 1f) {
                            count++;
                        }
                    }
                }

                return count == array.length;
            }
        }

        /**
         * Determine if the NArray in question is multi-hot (useful for BCE Loss)
         * @param input a {@link NArray} object
         * @return a boolean value of either true or false
         */
        private static boolean isMultiHot(NArray input) {
            boolean hasAtLeastOne = false;
            if (input.dimension == 1) {
                for (float element : input.getVecArray()) {
                    if (element != 1f && element != 0f) {
                        return false;
                    }

                    if (element == 1f) {
                        hasAtLeastOne = true;
                    }
                }
            } else {
                for (float[] row : input.getMatrixArray()) {
                    for (float element : row) {
                        if (element != 1f && element != 0f) {
                            return false;
                        }

                        if (element == 1f) {
                            hasAtLeastOne = true;
                        }
                    }
                }
            }

            return hasAtLeastOne;
        }

        /**
         * Check if each value in the provided {@link NArray} object is constrained between
         * the open interval of (0,1), as a result from the Sigmoid activation function.
         * @param input a {@link NArray} object
         * @return a boolean value of either true or false
         */
        private static boolean constrainedForBCE(NArray input) {
            if (input.dimension == 1) {
                for (float element : input.getVecArray()) {
                    if (element >= 1f ||  element <= 0f) {
                        return false;
                    }
                }

                return true;
            } else {
                for (float[] row : input.getMatrixArray()) {
                    for (float element : row) {
                        if (element >= 1f ||  element <= 0f) {
                            return false;
                        }
                    }
                }
                return true;
            }
        }
    }
}
