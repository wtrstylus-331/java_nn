package org.sdws.nn;

import org.sdws.mathematics.ActivationFunc;
import org.sdws.mathematics.LossAlgorithm;
import org.sdws.mathematics.NArray;
import org.sdws.mathematics.NNArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class NeuralNetwork {
    private final ArrayList<Layer> networkLayers = new ArrayList<>(1);
    private final ArrayList<NetworkLayer> layers;
    public Object output;
    public float loss;
    public float learningRate;
    public float accuracy;

    public NeuralNetwork(NetworkLayer... layers) {
        //this.networkLayers = new ArrayList<>(List.of(layers));
        this.layers = new ArrayList<>(List.of(layers));
        this.learningRate = 0.01f;
        this.loss = 0f;
        this.accuracy = 0f;
        this.output = null;
        for (Layer layer : this.networkLayers) {
            System.out.println(layer.toString());
        }
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
        this.loss = 0f;
        this.learningRate = 0.01f;
        this.accuracy = 0f;
    }

    /**
     * Performs one full pass of the provided {@code input} through the
     * entire neural network, storing the final output either as a {@code Float}
     * or {@link NNArray}, which can be accessed by referencing the {@code output}
     * attribute of the last layer in this network.
     * @param input the initial input to be taken by this network,
     *              represented as a {@link NNArray} object.<br><br>
     *              test
     */
    public void feedForward(NNArray input) {
        if (input == null) {
            return;
        }

        if (input.dimension == 1) { // output is either a vector or float
            NNArray prevOutput = input;
            for (Layer layer : this.networkLayers) {
                layer.ForwardPass(prevOutput);
                prevOutput = layer.postActivationOutput;
            }

            this.output = prevOutput;
        } else if (input.dimension == 2) { // output is an NNArray where each index represents output for each corresponding input index
            ArrayList<ArrayList<Float>> batch = (ArrayList<ArrayList<Float>>) input.nnarray();
            ArrayList<ArrayList<Float>> batchOutput = new ArrayList<>();

            for (ArrayList<Float> vec : batch) {
                NNArray prevOutput = NNArray.create(vec);
                for (Layer layer : this.networkLayers) {
                    layer.ForwardPass(prevOutput);
                    prevOutput = layer.postActivationOutput;
                }

                batchOutput.add((ArrayList<Float>) prevOutput.nnarray());
            }

            this.output = NNArray.create(batchOutput);
        }
    }

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
            case CrossCatEntropy -> calculateCCE(desiredOutput);
            case RootMeanSqErr -> calculateRMSE(desiredOutput);
            case BinaryCatEntropy -> calculateBCE(desiredOutput);
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
            case CrossCatEntropy -> calculateCCE(desiredOutput);
            case RootMeanSqErr -> calculateRMSE(desiredOutput);
            case BinaryCatEntropy -> calculateBCE(desiredOutput);
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

        } else {

        }
    }

    private void calculateCCE(Number desiredOutput) {
        if (!(this.output instanceof Number)) {
            throw new IllegalStateException("Cannot compute CCE loss when neural network output is not of a numeric type.");
        }

        if (!(desiredOutput.floatValue() == 1f) || !(desiredOutput.floatValue() == 0f)) {
            throw new IllegalArgumentException("Desired output must be either 0.0 or 1.0.");
        }

        if (((Number)this.output).getClass().equals(Float.class)) {
            this.loss = -desiredOutput.floatValue() * (float) Math.log((Float)this.output);
        } else if (((Number)this.output).getClass().equals(Double.class)) {
            this.loss = -desiredOutput.floatValue() * (float) Math.log((Float)this.output);
        } else if (((Number)this.output).getClass().equals(Integer.class)) {
            this.loss = -desiredOutput.intValue() * (float) Math.log((Float)this.output);
        } else if (((Number)this.output).getClass().equals(Long.class)) {
            this.loss = -desiredOutput.longValue() * (float) Math.log((Float)this.output);
        } else if (((Number)this.output).getClass().equals(Short.class)) {
            this.loss = -desiredOutput.shortValue() * (float) Math.log((Float)this.output);
        } else if (((Number)this.output).getClass().equals(Byte.class)) {
            this.loss = -desiredOutput.byteValue() * (float) Math.log((Float)this.output);
        } else {
            throw new IllegalArgumentException("Cannot compute CCE Loss of a non-numeric value.");
        }
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
                this.loss = (float) -Math.log(((NArray<Number>)this.output).get(index).floatValue());
            } else if (type.equals(Double.class)) {
                index = ((ArrayList<Number>) desiredOutput.innerArray()).indexOf((double)1);
                this.loss = (float) -Math.log(((NArray<Number>)this.output).get(index).doubleValue());
            } else if  (type.equals(Integer.class)) {
                index = ((ArrayList<Number>) desiredOutput.innerArray()).indexOf(1);
                this.loss = (float) -Math.log(((NArray<Number>)this.output).get(index).intValue());
            } else if (type.equals(Long.class)) {
                index = ((ArrayList<Number>) desiredOutput.innerArray()).indexOf((long)1);
                this.loss = (float) -Math.log(((NArray<Number>)this.output).get(index).longValue());
            } else if (type.equals(Short.class)) {
                index = ((ArrayList<Number>) desiredOutput.innerArray()).indexOf((short)1);
                this.loss = (float) -Math.log(((NArray<Number>)this.output).get(index).shortValue());
            } else if  (type.equals(Byte.class)) {
                index = ((ArrayList<Number>) desiredOutput.innerArray()).indexOf((byte)1);
                this.loss = (float) -Math.log(((NArray<Number>)this.output).get(index).byteValue());
            } else {
                throw new IllegalArgumentException("Cannot find one-hot element of a non-numeric type from desired output to calculate CCE loss.");
            }
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
                    clippedValue = Util.clip(((NArray<Number>)this.output).get(i, innerArray.get(i).indexOf((float)1)));
                    sum += (float) Math.log(clippedValue);
                } else if (type.equals(Integer.class)) {
                    clippedValue = Util.clip(((NArray<Number>) this.output).get(i, innerArray.get(i).indexOf(1)));
                    sum += (float) Math.log(clippedValue);
                } else if (type.equals(Long.class)) {
                    clippedValue = Util.clip(((NArray<Number>) this.output).get(i, innerArray.get(i).indexOf(1L)));
                    sum += (float) Math.log(clippedValue);
                } else if (type.equals(Short.class)) {
                        clippedValue = Util.clip(((NArray<Number>) this.output).get(i, innerArray.get(i).indexOf((short)1)));
                        sum += (float) Math.log(clippedValue);
                } else if (type.equals(Byte.class)) {
                    clippedValue = Util.clip(((NArray<Number>) this.output).get(i, innerArray.get(i).indexOf((byte)1)));
                    sum += (float) Math.log(clippedValue);
                } else {
                    throw new IllegalArgumentException("Cannot find one-hot element of a non-numeric type from desired output to calculate CCE loss.");
                }
            }

            this.loss = ((float) -1/n) * sum;
        }
    }


    private boolean validForCCE(NArray desiredOutput) {
        if (desiredOutput == null) {
            throw new IllegalArgumentException("Desired output for CCE/BCE loss must not be null.");
        }

        if (!Util.isOneHot(desiredOutput)) {
            throw new IllegalArgumentException("Vector/matrix desired output must be one-hot encoded.");
        }

        return true;
    }

    private boolean validForBCE(NArray desiredOutput) {
        if (desiredOutput == null) {
            throw new IllegalArgumentException("Desired output for CCE/BCE loss must not be null.");
        }

        if (!Objects.equals(this.layers.get(this.layers.size() - 1).activation, ActivationFunc.Sigmoid)) {
            throw new IllegalArgumentException(
                    "The layer being used to calculate BCE Loss must have its activation function set to the Sigmoid activation function (ActivationFunc.Sigmoid)."
            );
        }

        if (!Util.constrainedForBCE(this.layers.get(this.layers.size() - 1).postActivationOutput)) {
            throw new IllegalStateException(
                    "Outputs from Sigmoid activation function layer must be bounded between the open interval of (0,1)."
            );
        }

        if (!Util.isMultiHot(desiredOutput)) {
            throw new IllegalArgumentException("Vector/matrix desired output must be multi-hot encoded with either 1 or 0 as desired output values.");
        }

        return true;
    }

    private void calculateMSE(Number desiredOutput) {
        if (!(this.output instanceof Number)) {
            throw new IllegalStateException("Cannot compute MSE loss when neural network output is not of a numeric type.");
        }

        if (((Number)this.output).getClass().equals(Float.class)) {
            this.loss = (float)Math.pow((Float)this.output - desiredOutput.floatValue(), 2);
        } else if (((Number)this.output).getClass().equals(Double.class)) {
            this.loss = (float)Math.pow((Double) this.output - desiredOutput.doubleValue(), 2);
        } else if (((Number)this.output).getClass().equals(Integer.class)) {
            this.loss = (float)Math.pow((Integer) this.output - desiredOutput.intValue(), 2);
        }  else if (((Number)this.output).getClass().equals(Long.class)) {
            this.loss = (float)Math.pow((Long) this.output - desiredOutput.longValue(), 2);
        }  else if (((Number)this.output).getClass().equals(Short.class)) {
            this.loss = (float)Math.pow((Short) this.output - desiredOutput.shortValue(), 2);
        }  else if (((Number)this.output).getClass().equals(Byte.class)) {
            this.loss = (float)Math.pow((Byte) this.output - desiredOutput.byteValue(), 2);
        }
    }

    private void calculateMSE(NArray desiredOutput) {
        if (desiredOutput.dimension == 1) { // single output from single input
            if (!validforMSE(desiredOutput)) {
                return;
            }

            Class<?> type = desiredOutput.get(0).getClass();

            float n = desiredOutput.innerArray().size();
            float sum = 0f;

            for (int i = 0; i < n; i++) {
                if (type.equals(Integer.class)) {
                    sum += (float) Math.pow(((int)((NArray<?>) this.output).get(i) - (int)desiredOutput.get(i)), 2);
                } else if (type.equals(Float.class)) {
                    sum += (float) Math.pow(((float)((NArray<?>) this.output).get(i) - (float)desiredOutput.get(i)), 2);
                } else if (type.equals(Double.class)) {
                    sum += (float) Math.pow(((double)((NArray<?>) this.output).get(i) - (double)desiredOutput.get(i)), 2);
                } else if (type.equals(Long.class)) {
                    sum += (float) Math.pow(((long)((NArray<?>) this.output).get(i) - (long)desiredOutput.get(i)), 2);
                } else if (type.equals(Short.class)) {
                    sum += (float) Math.pow(((short)((NArray<?>) this.output).get(i) - (short)desiredOutput.get(i)), 2);
                } else if (type.equals(Byte.class)) {
                    sum += (float) Math.pow(((byte)((NArray<?>) this.output).get(i) - (byte)desiredOutput.get(i)), 2);
                } else {
                    throw new IllegalArgumentException("Cannot compute MSE of non-numeric type from desired output vector. (single output)");
                }
            }

            this.loss = sum / n;
        } else { // batch of outputs from batch of inputs
            if (!validforMSE(desiredOutput)) {
                return;
            }

            Class<?> type = desiredOutput.get(0, 0).getClass();

            float totalBatches = desiredOutput.innerArray().size();
            float innerTotal = ((ArrayList<Number>)desiredOutput.innerArray().get(0)).size();
            float sum;
            float batchSum = 0f;

            for (int i = 0; i < totalBatches; i++) {
                sum = 0f;

                for (int j = 0; j < innerTotal; j++) {
                    if (type.equals(Integer.class)) {
                        sum += (float) Math.pow(((int)((NArray<?>) this.output).get(i,j) - (int)desiredOutput.get(i,j)), 2);
                    } else if (type.equals(Float.class)) {
                        sum += (float) Math.pow(((float)((NArray<?>) this.output).get(i,j) - (float)desiredOutput.get(i,j)), 2);
                    } else if (type.equals(Double.class)) {
                        sum += (float) Math.pow(((double)((NArray<?>) this.output).get(i,j) - (double)desiredOutput.get(i,j)), 2);
                    } else if (type.equals(Long.class)) {
                        sum += (float) Math.pow(((long)((NArray<?>) this.output).get(i,j) - (long)desiredOutput.get(i,j)), 2);
                    } else if (type.equals(Short.class)) {
                        sum += (float) Math.pow(((short)((NArray<?>) this.output).get(i,j) - (short)desiredOutput.get(i,j)), 2);
                    } else if (type.equals(Byte.class)) {
                        sum += (float) Math.pow(((byte)((NArray<?>) this.output).get(i,j) - (byte)desiredOutput.get(i,j)), 2);
                    } else {
                        throw new IllegalArgumentException("Cannot compute MSE of non-numeric type from desired output matrix. (batch output)");
                    }
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

            Class<?> type = desiredOutput.get(0).getClass();

            float n = desiredOutput.innerArray().size();
            float sum = 0f;

            for (int i = 0; i < n; i++) {
                if (type.equals(Integer.class)) {
                    sum += (float) Math.pow(((int)((NArray<?>) this.output).get(i) - (int)desiredOutput.get(i)), 2);
                } else if (type.equals(Float.class)) {
                    sum += (float) Math.pow(((float)((NArray<?>) this.output).get(i) - (float)desiredOutput.get(i)), 2);
                } else if (type.equals(Double.class)) {
                    sum += (float) Math.pow(((double)((NArray<?>) this.output).get(i) - (double)desiredOutput.get(i)), 2);
                } else if (type.equals(Long.class)) {
                    sum += (float) Math.pow(((long)((NArray<?>) this.output).get(i) - (long)desiredOutput.get(i)), 2);
                } else if (type.equals(Short.class)) {
                    sum += (float) Math.pow(((short)((NArray<?>) this.output).get(i) - (short)desiredOutput.get(i)), 2);
                } else if (type.equals(Byte.class)) {
                    sum += (float) Math.pow(((byte)((NArray<?>) this.output).get(i) - (byte)desiredOutput.get(i)), 2);
                } else {
                    throw new IllegalArgumentException("Cannot compute MSE of non-numeric type from desired output vector. (single output)");
                }
            }

            this.loss = (float) Math.sqrt(sum / n);
        } else { // batch of outputs from batch of inputs
            if (!validforMSE(desiredOutput)) {
                return;
            }

            Class<?> type = desiredOutput.get(0, 0).getClass();

            float totalBatches = desiredOutput.innerArray().size();
            float innerTotal = ((ArrayList<Number>)desiredOutput.innerArray().get(0)).size();
            float sum;
            float batchSum = 0f;

            for (int i = 0; i < totalBatches; i++) {
                sum = 0f;

                for (int j = 0; j < innerTotal; j++) {
                    if (type.equals(Integer.class)) {
                        sum += (float) Math.pow(((int)((NArray<?>) this.output).get(i,j) - (int)desiredOutput.get(i,j)), 2);
                    } else if (type.equals(Float.class)) {
                        sum += (float) Math.pow(((float)((NArray<?>) this.output).get(i,j) - (float)desiredOutput.get(i,j)), 2);
                    } else if (type.equals(Double.class)) {
                        sum += (float) Math.pow(((double)((NArray<?>) this.output).get(i,j) - (double)desiredOutput.get(i,j)), 2);
                    } else if (type.equals(Long.class)) {
                        sum += (float) Math.pow(((long)((NArray<?>) this.output).get(i,j) - (long)desiredOutput.get(i,j)), 2);
                    } else if (type.equals(Short.class)) {
                        sum += (float) Math.pow(((short)((NArray<?>) this.output).get(i,j) - (short)desiredOutput.get(i,j)), 2);
                    } else if (type.equals(Byte.class)) {
                        sum += (float) Math.pow(((byte)((NArray<?>) this.output).get(i,j) - (byte)desiredOutput.get(i,j)), 2);
                    } else {
                        throw new IllegalArgumentException("Cannot compute MSE of non-numeric type from desired output matrix. (batch output)");
                    }
                }

                batchSum += (float) Math.sqrt(sum / innerTotal);
            }

            this.loss = (float) Math.sqrt(batchSum / totalBatches);
        }
    }

    private boolean validforMSE(NArray desiredOutput) {
        if (desiredOutput.dimension == 1) {
            if (desiredOutput.innerArray().isEmpty()) {
                throw new IllegalArgumentException("Desired output vector must not be empty. (single output)");
            }

            if (!Objects.equals(((NArray)this.output).shape, desiredOutput.shape)) {
                throw new IllegalArgumentException("Desired output vector must match the length of neural network output. (single output)");
            }
        } else {
            if (desiredOutput.innerArray().isEmpty() || ((ArrayList<?>)desiredOutput.innerArray().get(0)).isEmpty()) {
                throw new IllegalArgumentException("Desired output matrix must not be empty. (batch output)");
            }

            if (!Objects.equals(((NArray)this.output).shape, desiredOutput.shape)) {
                throw new IllegalArgumentException("Desired output matrix must match the length of neural network output. (batch output)");
            }
        }

        return true;
    }

//    /**
//     * Calculate Categorical Cross-Entropy Loss, to calculate loss of the neural network's
//     * {@code softmax} output.
//     * @param oneHotInput a strictly 1-dimensional {@link NNArray} object (vector)
//     */
//    public void calculateCCELoss(NNArray oneHotInput) {
//        if (oneHotInput == null) {
//            throw new IllegalArgumentException("Input for CCE must not be null");
//        }
//
//        if (oneHotInput.dimension >= 3) {
//            throw new IllegalArgumentException("Input dimensions for CCE does not match network output dimension.");
//        }
//
//        if (!Util.isOneHot(oneHotInput)) {
//            throw new IllegalArgumentException("Input for CCE is not in the form of one-hot input vector.");
//        }
//
//        if (this.output.dimension == 1) { // single one hot vec
//            if (!Objects.equals(this.output.shape.get(0), oneHotInput.shape.get(0))) {
//                throw new IllegalArgumentException("One-hot vector length for CCE is not equal to network output.");
//            }
//
//            int index = ((ArrayList<Float>) oneHotInput.nnarray()).indexOf((float)1);
//            this.loss = (float) -Math.log(this.output.get(index));
//
//        } else { // batches of one hot vecs for batches of outputs
//            if (oneHotInput.dimension == 1) {
//                throw new IllegalArgumentException("One-hot desired output for CCE is of the wrong dimension. (Must be a matrix with rows containing one-hot vectors corresponding to output from input batch)");
//            }
//
//            boolean equalRows = Objects.equals(this.output.shape.get(0), oneHotInput.shape.get(0));
//            boolean equalCols = Objects.equals(this.output.shape.get(1), oneHotInput.shape.get(1));
//            if (!equalRows && !equalCols) {
//                throw new IllegalArgumentException("One-hot vector length for CCE is not equal to network batch output matrix.");
//            }
//
//            float sum = 0f;
//            int n = this.output.shape.get(0);
//            ArrayList<ArrayList<Float>> innerArray = (ArrayList<ArrayList<Float>>) oneHotInput.nnarray();
//            for (int i = 0; i < n; i++) {
//                float clippedValue = Util.clip(this.output.get(i, innerArray.get(i).indexOf((float)1)));
//                //sum += (float) Math.log(this.output.get(i, innerArray.get(i).indexOf((float)1)));
//                sum += (float) Math.log(clippedValue);
//            }
//
//            this.loss = ((float) -1/n) * sum;
//        }
//    }

//    /**
//     * Calculate Binary Cross-Entropy Loss, to calculate loss of the neural network's
//     * {@code softmax} output.
//     * @param oneHotInput a strictly 1-dimensional {@link NNArray} object (vector)
//     */
//    public void calculateBCELoss(NNArray oneHotInput) {
//        if (oneHotInput == null) {
//            throw new IllegalArgumentException("Input for BCE must not be null");
//        }
//
//        if (oneHotInput.dimension > 3) {
//            throw new IllegalArgumentException("Input dimensions for BCE does not match network output dimension.");
//        }
//
//        if (!Util.isOneHot(oneHotInput)) {
//            throw new IllegalArgumentException("Input for BCE is not in the form of one-hot input vector.");
//        }
//
//        if (oneHotInput.dimension == 1) {
//            if (!(oneHotInput.shape.get(1) == 2)) {
//                throw new IllegalArgumentException("One-hot vector length for BCE should be two elements.");
//            }
//        }
//
//        if (this.output.dimension == 1) { // single binary one hot vec
//
//        } else { // batches of binary one hot vecs for batches of binary outputs
//
//        }
//    }

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
        for (int i = this.networkLayers.size() - 1; i >= 0; i--) {

        }
    }

    private static class Util {
//        private static Float getMax(ArrayList<Float> array) {
//            float max = Float.NEGATIVE_INFINITY;
//            for (Float element : array) {
//                if (element > max) {
//                    max = element;
//                }
//            }
//            return max;
//        }

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
         * represented by the {@link NNArray} object is a one-hot vector or not.
         * @param input a {@link NNArray} object representing a vector
         * @return a boolean value of either {@code true} or {@code false}
         */
        private static boolean isOneHot(NNArray input) {
            if (input.dimension == 1) {
                int count = 0;
                for (Float element : (ArrayList<Float>) input.nnarray()) {
                    if (element == (float) 1) {
                        count++;
                    }
                }

                return count == 1;
            } else {
                int count = 0;
                for (ArrayList<Float> row : (ArrayList<ArrayList<Float>>) input.nnarray()) {
                    for (Float element : row) {
                        if (element == (float) 1) {
                            count++;
                        }
                    }
                }

                return count == input.shape.get(0);
            }
        }

        private static boolean isOneHot(NArray input) {
            if (input.dimension == 1) {
                int count = 0;
                for (int i = 0; i < (int) input.shape.get(0); i++) {
                    Class<?> type = input.get(0).getClass();

                    if (type.equals(Float.class)) {
                        if (input.get(i).floatValue() == 1f) {
                            count++;
                        }
                    } else if (type.equals(Double.class)) {
                        if (input.get(i).doubleValue() == 1d) {
                            count++;
                        }
                    } else if (type.equals(Integer.class)) {
                        if (input.get(i).intValue() == 1) {
                            count++;
                        }
                    } else if (type.equals(Long.class)) {
                        if (input.get(i).longValue() == 1L) {
                            count++;
                        }
                    } else if (type.equals(Short.class)) {
                        if (input.get(i).shortValue() == (short) 1) {
                            count++;
                        }
                    } else if (type.equals(Byte.class)) {
                        if (input.get(i).byteValue() == (byte) 1) {
                            count++;
                        }
                    }
                }

                return count == 1;
            } else {
                int count = 0;
                for (int i = 0; i < (int) input.shape.get(0); i++) {
                    for (int j = 1; j < (int) input.shape.get(1); j++) {
                        Class<?> type = input.get(i, j).getClass();

                        if (type.equals(Float.class)) {
                            if (input.get(i).floatValue() == 1f) {
                                count++;
                            }
                        } else if (type.equals(Double.class)) {
                            if (input.get(i).doubleValue() == 1d) {
                                count++;
                            }
                        } else if (type.equals(Integer.class)) {
                            if (input.get(i).intValue() == 1) {
                                count++;
                            }
                        } else if (type.equals(Long.class)) {
                            if (input.get(i).longValue() == 1L) {
                                count++;
                            }
                        } else if (type.equals(Short.class)) {
                            if (input.get(i).shortValue() == (short) 1) {
                                count++;
                            }
                        } else if (type.equals(Byte.class)) {
                            if (input.get(i).byteValue() == (byte) 1) {
                                count++;
                            }
                        }
                    }
                }

                return count == (int) input.shape.get(0);
            }
        }

        /**
         * Determine if the NArray in question is multi-hot (useful for BCE Loss)
         * @param input a {@link NArray} object
         * @return a boolean value of either true or false
         */
        private static boolean isMultiHot(NArray input) {
            if (input.dimension == 1) {
                for (Number element : (ArrayList<Number>) input.innerArray()) {
                    if (!Objects.equals(element.floatValue(), 1f) || !Objects.equals(element.floatValue(), 0f)) {
                        return false;
                    }
                }

                return true;
            } else {
                for (ArrayList<Number> row : (ArrayList<ArrayList<Number>>) input.innerArray()) {
                    for (Number element : row) {
                        if (!Objects.equals(element.floatValue(), 1f) || !Objects.equals(element.floatValue(), 0f)) {
                            return false;
                        }
                    }
                }
                return true;
            }
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
