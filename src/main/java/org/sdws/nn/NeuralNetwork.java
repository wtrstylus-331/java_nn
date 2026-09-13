package org.sdws.nn;

import org.sdws.util.ActivationFunc;
import org.sdws.util.LossAlgorithm;
import org.sdws.mathematics.NArray;
import org.sdws.mathematics.NNArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class NeuralNetwork {
    //private final ArrayList<Layer> networkLayers = new ArrayList<>(1);
    private final ArrayList<NetworkLayer> layers;
    public Object output;
    public float loss;
    public float learningRate;
    public float accuracy;

    public NeuralNetwork(NetworkLayer... layers) {
        //this.networkLayers = new ArrayList<>(List.of(layers));
        this.layers = new ArrayList<>(List.of(layers));
        this.learningRate = 0.01f;
        this.loss = Float.NaN;
        this.accuracy = 0f;
        this.output = null;
//        for (Layer layer : this.networkLayers) {
//            System.out.println(layer.toString());
//        }
    }

    public void feedForward(NArray input) {
        if (input == null) {
            return;
        }

        if (input.dimension == 1) {
            NArray prevOutput = input;
            for (NetworkLayer layer : this.layers) {
                layer.ForwardPass(prevOutput);
                prevOutput = layer.postActivationOutput;
            }

            this.output = prevOutput;
        } else if (input.dimension == 2) {
            ArrayList<ArrayList<Number>> batch = (ArrayList<ArrayList<Number>>) input.innerArray();
            ArrayList<ArrayList<Number>> batchOutput = new ArrayList<>();

            for (ArrayList<Number> vec : batch) {
                NArray prevOutput = NArray.FromCollection(vec);
                for (NetworkLayer layer : this.layers) {
                    layer.ForwardPass(prevOutput);
                    prevOutput = layer.postActivationOutput;
                }

                batchOutput.add((ArrayList<Number>) prevOutput.innerArray());
            }

            this.output = NArray.FromNested(batchOutput);
        }
    }

    public void erase() {
        this.layers.clear();
        this.output = null;
        this.loss = Float.NaN;
        this.learningRate = 0.01f;
        this.accuracy = 0f;
    }

//    /**
//     * Performs one full pass of the provided {@code input} through the
//     * entire neural network, storing the final output either as a {@code Float}
//     * or {@link NNArray}, which can be accessed by referencing the {@code output}
//     * attribute of the last layer in this network.
//     * @param input the initial input to be taken by this network,
//     *              represented as a {@link NNArray} object.<br><br>
//     *              test
//     */
//    public void feedForward(NNArray input) {
//        if (input == null) {
//            return;
//        }
//
//        if (input.dimension == 1) { // output is either a vector or float
//            NNArray prevOutput = input;
//            for (Layer layer : this.networkLayers) {
//                layer.ForwardPass(prevOutput);
//                prevOutput = layer.postActivationOutput;
//            }
//
//            this.output = prevOutput;
//        } else if (input.dimension == 2) { // output is an NNArray where each index represents output for each corresponding input index
//            ArrayList<ArrayList<Float>> batch = (ArrayList<ArrayList<Float>>) input.nnarray();
//            ArrayList<ArrayList<Float>> batchOutput = new ArrayList<>();
//
//            for (ArrayList<Float> vec : batch) {
//                NNArray prevOutput = NNArray.create(vec);
//                for (Layer layer : this.networkLayers) {
//                    layer.ForwardPass(prevOutput);
//                    prevOutput = layer.postActivationOutput;
//                }
//
//                batchOutput.add((ArrayList<Float>) prevOutput.nnarray());
//            }
//
//            this.output = NNArray.create(batchOutput);
//        }
//    }

//    /**
//     * Returns the final output computed after the forward pass has been called for
//     * the neural network.
//     * @return a {@link NNArray} object that can represent either one of the following based
//     * on the implementation of the neural network:<br><br>
//     * - a 2-dimensional array in the instance of forward passing batches of vectors (1-dimensional arrays)
//     * into the neural network. Each index in the output {@link NNArray} corresponds to the index in the
//     * batch argument, a 2-dimensional array<br><br>
//     * - a 1-dimensional array representing a vector output after an entire forward pass from a single
//     * vector input of a {@link NNArray}<br><br>
//     * - an array with dimension and shape both being exactly {@code 1}, which in this instance can be
//     * represented as a {@code [x]}, where {@code x} is any real number of type {@link Float}
//     */
//    public NNArray output() {
//        if (this.networkLayers.isEmpty()) {
//            return null;
//        } else {
//            return (NNArray) this.output;
//            //return this.networkLayers.get(this.networkLayers.size() - 1).output;
//        }
//    }

    /**
     * Calculate the loss of the neural network based on vector outputs from single inputs, or
     * based on matrix outputs as part of batches of vector inputs represented as matrices.
     * @param desiredOutput
     * @param lossAlgorithm the algorithm value from {@link LossAlgorithm}
     */
    public void calculateLoss(NArray desiredOutput, LossAlgorithm lossAlgorithm) {
        if (desiredOutput == null || desiredOutput.dimension > 2 || lossAlgorithm == null) {
            throw new IllegalArgumentException("Desired output array must not be null to calculate loss.");
        }

        switch (lossAlgorithm) {
            case CCELoss -> calculateCCE(desiredOutput);
            case RMSELoss -> calculateRMSE(desiredOutput);
            case BCELoss -> calculateBCE(desiredOutput);
            default -> calculateMSE(desiredOutput);
        }

        //this.calculateAccuracy(desiredOutput, lossAlgorithm);
    }

    /**
     * Intended to calculate loss solely in cases where the neural network produces a single float
     * output (1 neuron in the last layer).
     * @param desiredOutput
     * @param lossAlgorithm
     */
    public void calculateLoss(Number desiredOutput, LossAlgorithm lossAlgorithm) {
        if (desiredOutput == null || lossAlgorithm == null) {
            throw new IllegalArgumentException("Desired output array must not be null to calculate loss.");
        }

        switch (lossAlgorithm) {
            case CCELoss -> calculateCCE(desiredOutput);
            case RMSELoss -> calculateRMSE(desiredOutput);
            case BCELoss -> calculateBCE(desiredOutput);
            default -> calculateMSE(desiredOutput);
        }
    }

    private void calculateBCE(Number desiredOutput) {
        if (!(this.output instanceof Number)) {
            throw new IllegalStateException("Cannot compute BCE loss when neural network output is not of a numeric type.");
        }

        if (!(desiredOutput.floatValue() == 1f) || !(desiredOutput.floatValue() == 0f)) {
            throw new IllegalArgumentException("Desired output must be either 0.0 or 1.0.");
        }

        this.loss = desiredOutput.floatValue() * (float)Math.log(((Number) this.output).floatValue()) +
                (1 - desiredOutput.floatValue()) * (float)Math.log(1 - ((Number) this.output).floatValue());

//        if (((Number)this.output).getClass().equals(Float.class)) {
//            this.loss = desiredOutput.floatValue() *
//                    (float)Math.log(this.loss) + (1 - desiredOutput.floatValue()) * (float) Math.log(1 - this.loss);
//        } else if (((Number)this.output).getClass().equals(Double.class)) {
//            this.loss = (float) (desiredOutput.doubleValue() *
//                    Math.log(this.loss) + (1 - desiredOutput.doubleValue()) * Math.log(1 - this.loss));
//        } else if (((Number)this.output).getClass().equals(Integer.class)) {
//            this.loss = (desiredOutput.intValue() *
//                    (float)Math.log(this.loss) + (1 - desiredOutput.intValue()) * (float)Math.log(1 - this.loss));
//        } else if (((Number)this.output).getClass().equals(Long.class)) {
//            this.loss = (desiredOutput.longValue() *
//                    (float)Math.log(this.loss) + (1 - desiredOutput.longValue()) * (float)Math.log(1 - this.loss));
//        } else if (((Number)this.output).getClass().equals(Short.class)) {
//            this.loss = (desiredOutput.shortValue() *
//                    (float)Math.log(this.loss) + (1 - desiredOutput.shortValue()) * (float)Math.log(1 - this.loss));
//        } else if (((Number)this.output).getClass().equals(Byte.class)) {
//            this.loss = (desiredOutput.byteValue() *
//                    (float)Math.log(this.loss) + (1 - desiredOutput.byteValue()) * (float)Math.log(1 - this.loss));
//        } else {
//            throw new IllegalArgumentException("Cannot compute BCE Loss of a non-numeric value from the actual binary label argument.");
//        }
    }

    private void calculateBCE(NArray desiredOutput) {
        if (!validForBCE(desiredOutput)) {
            return;
        }

        if (desiredOutput.dimension == 1) {
            if (!(((NArray) this.output).dimension == 1)) {
                throw new IllegalArgumentException("Cannot calculate loss from output matrix when desired output is a vector.");
            }

            int n = desiredOutput.innerArray().size();
            float total = 0f;

            for (int i = 1; i < n; i++) {
                total += desiredOutput.get(i).floatValue() * (float)Math.log(((NArray<Number>)this.output).get(i).floatValue()) +
                        (1 - desiredOutput.get(i).floatValue()) * (float)Math.log(1 - ((NArray<Number>)this.output).get(i).floatValue());
            }

            this.loss = (-1f / n) * total;
        } else {
            if (!(((NArray) this.output).dimension == 2)) {
                throw new IllegalArgumentException("Cannot calculate loss from output vector when desired output is a matrix representing batches of outputs.");
            }

            int n = (int)desiredOutput.shape.get(1);
            float batchesTotal = 0f;

            for (int i = 1; i < desiredOutput.innerArray().size(); i++) {
                float batchLoss = 0f;

                for (int j = 1; j < n; j++) {
                    batchLoss += desiredOutput.get(i,j).floatValue() * (float)Math.log(((NArray<Number>)this.output).get(i,j).floatValue()) +
                            (1 - desiredOutput.get(i,j).floatValue()) * (float)Math.log(1 - ((NArray<Number>)this.output).get(i,j).floatValue());
                }

                batchesTotal += (-1f / n) * batchLoss;
            }

            this.loss = batchesTotal / ((float) desiredOutput.innerArray().size());
        }
    }

    private void calculateCCE(Number desiredOutput) {
        if (!(this.output instanceof Number)) {
            throw new IllegalStateException("Cannot compute CCE loss when neural network output is not of a numeric type.");
        }

        if (!(desiredOutput.floatValue() == 1f) || !(desiredOutput.floatValue() == 0f)) {
            throw new IllegalArgumentException("Desired output must be either 0.0 or 1.0.");
        }

        this.loss = -desiredOutput.floatValue() * (float) Math.log((Float)this.output);

//        if (((Number)this.output).getClass().equals(Float.class)) {
//            this.loss = -desiredOutput.floatValue() * (float) Math.log((Float)this.output);
//        } else if (((Number)this.output).getClass().equals(Double.class)) {
//            this.loss = -desiredOutput.floatValue() * (float) Math.log((Float)this.output);
//        } else if (((Number)this.output).getClass().equals(Integer.class)) {
//            this.loss = -desiredOutput.intValue() * (float) Math.log((Float)this.output);
//        } else if (((Number)this.output).getClass().equals(Long.class)) {
//            this.loss = -desiredOutput.longValue() * (float) Math.log((Float)this.output);
//        } else if (((Number)this.output).getClass().equals(Short.class)) {
//            this.loss = -desiredOutput.shortValue() * (float) Math.log((Float)this.output);
//        } else if (((Number)this.output).getClass().equals(Byte.class)) {
//            this.loss = -desiredOutput.byteValue() * (float) Math.log((Float)this.output);
//        } else {
//            throw new IllegalArgumentException("Cannot compute CCE Loss of a non-numeric value.");
//        }
    }

    private void calculateCCE(NArray desiredOutput) {
        if (!validForCCE(desiredOutput)) {
            return;
        }

        if (desiredOutput.dimension == 1) {
            if (!Objects.equals(((NArray) this.output).shape.get(1), desiredOutput.shape.get(1))) {
                throw new IllegalArgumentException("One-hot vector length for CCE is not equal to network output.");
            }

            Class<?> type = desiredOutput.innerArray().get(0).getClass();
            int index;

            if (type.equals(Float.class)) {
                index = ((ArrayList<Number>) desiredOutput.innerArray()).indexOf((float)1);
            } else if (type.equals(Double.class)) {
                index = ((ArrayList<Number>) desiredOutput.innerArray()).indexOf((double)1);
            } else if  (type.equals(Integer.class)) {
                index = ((ArrayList<Number>) desiredOutput.innerArray()).indexOf(1);
            } else if (type.equals(Long.class)) {
                index = ((ArrayList<Number>) desiredOutput.innerArray()).indexOf((long)1);
            } else if (type.equals(Short.class)) {
                index = ((ArrayList<Number>) desiredOutput.innerArray()).indexOf((short)1);
            } else if  (type.equals(Byte.class)) {
                index = ((ArrayList<Number>) desiredOutput.innerArray()).indexOf((byte)1);
            } else {
                throw new IllegalArgumentException("Cannot find one-hot element of a non-numeric type from desired output to calculate CCE loss.");
            }

            this.loss = (float) -Math.log(((NArray<Number>)this.output).get(index).floatValue());
        } else {
            if (desiredOutput.dimension == 1) {
                throw new IllegalArgumentException("One-hot desired output for CCE is of the wrong dimension. (Must be a matrix with rows containing one-hot vectors corresponding to output from input batch)");
            }

            boolean equalRows = Objects.equals(((NArray)this.output).shape.get(0), desiredOutput.shape.get(0));
            boolean equalCols = Objects.equals(((NArray)this.output).shape.get(1), desiredOutput.shape.get(1));
            if (!equalRows && !equalCols) {
                throw new IllegalArgumentException("One-hot matrix batch size for CCE is not equal to network batch output matrix.");
            }

            float sum = 0f;
            int n = (((NArray<Number>)this.output).shape.get(0).intValue());
            ArrayList<ArrayList<Number>> innerArray = (ArrayList<ArrayList<Number>>) desiredOutput.innerArray();
            Class<?> type = ((ArrayList<Number>)desiredOutput.innerArray().get(0)).get(0).getClass();

            for (int i = 0; i < n; i++) {
                float clippedValue;

                if (type.equals(Float.class) ||  type.equals(Double.class)) {
                    clippedValue = Util.clip(((NArray<Number>)this.output).get(i, innerArray.get(i).indexOf((float)1)).floatValue());
                } else if (type.equals(Integer.class)) {
                    clippedValue = Util.clip(((NArray<Number>) this.output).get(i, innerArray.get(i).indexOf(1)).floatValue());
                } else if (type.equals(Long.class)) {
                    clippedValue = Util.clip(((NArray<Number>) this.output).get(i, innerArray.get(i).indexOf(1L)).floatValue());
                } else if (type.equals(Short.class)) {
                    clippedValue = Util.clip(((NArray<Number>) this.output).get(i, innerArray.get(i).indexOf((short)1)).floatValue());
                } else if (type.equals(Byte.class)) {
                    clippedValue = Util.clip(((NArray<Number>) this.output).get(i, innerArray.get(i).indexOf((byte)1)).floatValue());
                } else {
                    throw new IllegalArgumentException("Cannot find one-hot element of a non-numeric type from desired output to calculate CCE loss.");
                }

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

        if (!Objects.equals(this.layers.get(this.layers.size() - 1).activation, ActivationFunc.Sigmoid)) {
            throw new IllegalArgumentException(
                    "The layer being used to calculate BCE Loss must have its activation function set to the Sigmoid activation function (ActivationFunc.Sigmoid)."
            );
        }

        if (!Util.constrainedForBCE(this.layers.get(this.layers.size() - 1).postActivationOutput)) {
            throw new IllegalStateException(
                    "Outputs from Sigmoid activation function layer must be bounded between the open interval of (0,1) for BCE loss calculation."
            );
        }

        if (!Util.isMultiHot(desiredOutput)) {
            throw new IllegalArgumentException("Vector/matrix desired output must be multi-hot encoded with either 1 or 0 as desired output values for BCE loss.");
        }

        return true;
    }

    private void calculateMSE(Number desiredOutput) {
        if (!(this.output instanceof Number)) {
            throw new IllegalStateException("Cannot compute MSE loss when neural network output is not of a numeric type.");
        }

        this.loss = (float)Math.pow((Float)this.output - desiredOutput.floatValue(), 2);
    }

    private void calculateMSE(NArray desiredOutput) {
        if (desiredOutput.dimension == 1) { // single output from single input
            if (!validforMSE(desiredOutput)) {
                return;
            }

            float n = desiredOutput.innerArray().size();
            float sum = 0f;

            for (int i = 0; i < n; i++) {
                sum += (float) Math.pow(((float)((NArray<?>) this.output).get(i) - (float)desiredOutput.get(i)), 2);
            }

            this.loss = sum / n;
        } else { // batch of outputs from batch of inputs
            if (!validforMSE(desiredOutput)) {
                return;
            }

            float totalBatches = desiredOutput.innerArray().size();
            float innerTotal = ((ArrayList<Number>)desiredOutput.innerArray().get(0)).size();
            float sum;
            float batchSum = 0f;

            for (int i = 0; i < totalBatches; i++) {
                sum = 0f;

                for (int j = 0; j < innerTotal; j++) {
                    sum += (float) Math.pow(((float)((NArray<?>) this.output).get(i,j) - (float)desiredOutput.get(i,j)), 2);
                }

                batchSum += sum / innerTotal;
            }

            this.loss = batchSum / totalBatches;
        }
    }

    private void calculateRMSE(Number desiredOutput) {
        calculateMSE(desiredOutput);
        this.loss = (float) Math.sqrt(this.loss);
    }

    private void calculateRMSE(NArray desiredOutput) {
        if (desiredOutput.dimension == 1) { // single output from single input
            if (!validforMSE(desiredOutput)) {
                return;
            }

            float n = desiredOutput.innerArray().size();
            float sum = 0f;

            for (int i = 0; i < n; i++) {
                sum += (float) Math.pow(((float)((NArray<?>) this.output).get(i) - (float)desiredOutput.get(i)), 2);
            }

            this.loss = (float) Math.sqrt(sum / n);
        } else { // batch of outputs from batch of inputs
            if (!validforMSE(desiredOutput)) {
                return;
            }

            float totalBatches = desiredOutput.innerArray().size();
            float innerTotal = ((ArrayList<Number>)desiredOutput.innerArray().get(0)).size();
            float sum;
            float batchSum = 0f;

            for (int i = 0; i < totalBatches; i++) {
                sum = 0f;

                for (int j = 0; j < innerTotal; j++) {
                    sum += (float) Math.pow(((float)((NArray<?>) this.output).get(i,j) - (float)desiredOutput.get(i,j)), 2);
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
            if (desiredOutput.innerArray().isEmpty()) {
                throw new IllegalArgumentException("Desired output vector must not be empty for MSE loss. (single output)");
            }

            if (!Objects.equals(((NArray)this.output).shape, desiredOutput.shape)) {
                throw new IllegalArgumentException("Desired output vector must match the length of neural network output for MSE loss. (single output)");
            }
        } else {
            if (desiredOutput.innerArray().isEmpty() || ((ArrayList<?>)desiredOutput.innerArray().get(0)).isEmpty()) {
                throw new IllegalArgumentException("Desired output matrix must not be empty for MSE loss. (batch output)");
            }

            if (!Objects.equals(((NArray)this.output).shape, desiredOutput.shape)) {
                throw new IllegalArgumentException("Desired output matrix must match the length of neural network output for MSE loss. (batch output)");
            }
        }

        return true;
    }

//

//    public Float calculateAccuracy(NNArray oneHotInput) {
//        if (oneHotInput == null) {
//            throw new IllegalArgumentException("Input for CCE must not be null");
//        }
//
//        if (oneHotInput.dimension > 3) {
//            throw new IllegalArgumentException("Input dimensions for CCE does not match network output dimension.");
//        }
//
//        if (!Util.isOneHot(oneHotInput)) {
//            throw new IllegalArgumentException("Input for CCE is not in the form of one-hot input vector.");
//        }
//
//        if (this.output.dimension == 1) {
//            int desiredIndex = oneHotInput.nnarray().indexOf((float)1);
//            int actualIndex = 0;
//            float max = Float.NEGATIVE_INFINITY;
//
//            // assumption that output is formatted into distribution from cce
//            for (int i = 0; i < this.output.nnarray().size(); i++) {
//                if (this.output.get(i) > max) {
//                    actualIndex = i;
//                    max = this.output.get(i);
//                }
//            }
//
//            if (Objects.equals(actualIndex, desiredIndex)) {
//                return 1f;
//            } else {
//                return 0f;
//            }
//        } else {
//            int batches = this.output.shape.get(0);
//            int correct = 0;
//
//            for (int i = 0; i < this.output.shape.get(0); i++) {
//                int desiredIndex = ((ArrayList<Float>)oneHotInput.nnarray().get(i)).indexOf((float)1);
//                float max = Util.getMax((ArrayList<Float>) this.output.nnarray().get(i));
//                if (((ArrayList<Float>) this.output.nnarray().get(i)).indexOf(max) == desiredIndex) {
//                    correct++;
//                }
//            }
//
//            return (float) correct / (float) batches;
//        }
//    }

//    private void accuracyFromCCE(NArray desiredOutput) {
//        if (((NArray)this.output).dimension == 1) {
//            Class<?> type = desiredOutput.get(0).getClass();
//            int desiredIndex;
//
//            if (type.equals(Float.class)) {
//                desiredIndex = desiredOutput.innerArray().indexOf((float)1);
//            } else if (type.equals(Double.class)) {
//                desiredIndex = desiredOutput.innerArray().indexOf((double) 1);
//            } else if (type.equals(Integer.class)) {
//                desiredIndex = desiredOutput.innerArray().indexOf(1);
//            } else if (type.equals(Long.class)) {
//                desiredIndex = desiredOutput.innerArray().indexOf(1L);
//            } else if (type.equals(Short.class)) {
//                desiredIndex = desiredOutput.innerArray().indexOf((short)1);
//            } else if (type.equals(Byte.class)) {
//                desiredIndex = desiredOutput.innerArray().indexOf((byte)1);
//            } else {
//                throw new IllegalArgumentException("Desired index for loss calculation (CCE) cannot be parsed from non-numeric type.");
//            }
//
//            int actualIndex = 0;
//            float max = Float.NEGATIVE_INFINITY;
//
//            for (int i = 0; i < ((NArray)this.output).innerArray().size(); i++) {
//                if (((NArray)this.output).get(i).floatValue() > max) {
//                    actualIndex = i;
//                    max = ((NArray)this.output).get(i).floatValue();
//                }
//            }
//
//            if (Objects.equals(actualIndex, desiredIndex)) {
//                this.accuracy = 1f;
//            } else {
//                this.accuracy = 0f;
//            }
//        } else {
//            int batches = (int)((NArray)this.output).shape.get(0);
//            int correct = 0;
//
//            for (int i = 0; i < (int)((NArray)this.output).shape.get(0); i++) {
//                int desiredIndex = ((ArrayList<Number>)desiredOutput.innerArray().get(i)).indexOf((float)1);
//
//                float max = Util.getMax((ArrayList<Number>) ((NArray)this.output).innerArray().get(i));
//                if (((ArrayList<Number>) ((NArray)this.output).innerArray().get(i)).indexOf(max) == desiredIndex) {
//                    correct++;
//                }
//            }
//
//            this.accuracy =  (float) correct / (float) batches;
//        }
//    }
//
//    private void accuracyFromBCE(NArray desiredOutput) {
//
//    }
//
//    private void accuracyFromRMSE(NArray desiredOutput) {
//
//    }
//
//    private void accuracyFromMSE(NArray desiredOutput) {
//
//    }
//
//    private void calculateAccuracy(NArray desiredOutput, LossAlgorithm lossAlg) {
//        switch (lossAlg) {
//            case CrossCatEntropy -> accuracyFromCCE(desiredOutput);
//            case BinaryCatEntropy -> accuracyFromBCE(desiredOutput);
//            case RootMeanSqErr ->  accuracyFromRMSE(desiredOutput);
//            default -> accuracyFromMSE(desiredOutput);
//        }
//    }

    private void calculateGradients() {
        for (int i = this.layers.size() - 1; i >= 0; i--) {

        }
    }

    private static class Util {
        private static Float getMax(ArrayList<Number> array) {
            float max = Float.NEGATIVE_INFINITY;
            Class<?> type = array.get(0).getClass();

            for (int i = 0; i < array.size(); i++) {
                if (type.equals(Float.class) || type.equals(Double.class)) {
                    if (array.get(i).floatValue() > max) {
                        max = array.get(i).floatValue();
                    }
                } else if (type.equals(Integer.class)) {
                    if (array.get(i).intValue() > max) {
                        max = array.get(i).intValue();
                    }
                } else if (type.equals(Long.class)) {
                    if (array.get(i).longValue() > max) {
                        max = array.get(i).longValue();
                    }
                } else if (type.equals(Short.class)) {
                    if (array.get(i).shortValue() > max) {
                        max = array.get(i).shortValue();
                    }
                } else if (type.equals(Byte.class)) {
                    if (array.get(i).byteValue() > max) {
                        max = array.get(i).byteValue();
                    }
                } else {
                    throw new IllegalArgumentException("Unable to find max element for batch loss computation (CCE).");
                }
            }

            return max;
        }

        private static Float clip(float value) {
            if (value < 1e-7f) {
                return 1e-7f;
            }  else if (value > 1f - 1e-7f) {
                return 1f-1e-7f;
            }
            return value;
        }

        private static Double clip(double value) {
            if (value < 1e-7d) {
                return 1e-7d;
            }  else if (value > 1d - 1e-7d) {
                return 1d-1e-7d;
            }
            return value;
        }

        private static Float clip(Number value) {
            if (value.floatValue() < 1e-7f) {
                return 1e-7f;
            }  else if (value.floatValue() > 1f - 1e-7f) {
                return 1f-1e-7f;
            }
            return value.floatValue();
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
                for (int i = 0; i < input.innerArray().size(); i++) {
                    if (input.get(i).floatValue() == 1f) {
                        count++;
                    }
                }

                return count == 1;
            } else {
                for (int i = 0; i < input.innerArray().size(); i++) {
                    for (int j = 0; j < ((ArrayList<Number>)input.innerArray().get(i)).size(); j++) {
                        if (input.get(i,j).floatValue() == 1f) {
                            count++;
                        }
                    }
                }

                return count == input.innerArray().size();
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
                for (Number element : (ArrayList<Number>) input.innerArray()) {
                    if (element.floatValue() != 1f && element.floatValue() != 0f) {
                        return false;
                    }

                    if (element.floatValue() == 1f) {
                        hasAtLeastOne = true;
                    }
                }
            } else {
                for (ArrayList<Number> row : (ArrayList<ArrayList<Number>>) input.innerArray()) {
                    for (Number element : row) {
                        if (element.floatValue() != 1f && element.floatValue() != 0f) {
                            return false;
                        }

                        if (element.floatValue() == 1f) {
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
                for (Number element : (ArrayList<Number>) input.innerArray()) {
                    if (element.floatValue() >= 1f ||  element.floatValue() <= 0f) {
                        return false;
                    }
                }

                return true;
            } else {
                for (ArrayList<Number> row : (ArrayList<ArrayList<Number>>) input.innerArray()) {
                    for (Number element : row) {
                        if (element.floatValue() >= 1f ||  element.floatValue() <= 0f) {
                            return false;
                        }
                    }
                }
                return true;
            }
        }
    }
}
