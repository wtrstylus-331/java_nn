package org.sdws.mathematics;

import org.sdws.util.ActivationFunc;

public class Activation {
    /**
     * ReLU stands for {@link <a href="https://en.wikipedia.org/wiki/Rectified_linear_unit">'Rectified Linear Unit'</a>},
     * which can also be expressed as:
     * <p><code>Math.max(0, x)</code> for any real number <code>x</code>. <p>This function mutates
     * an input matrix in place by applying the ReLU function to each value in the matrix.
     * @param input a 'NArray' object
     */
    public static void ReLU(NArray input) {
        if (input.dimension == 1) {
            mutateVectorActivation(input, ActivationFunc.ReLU);
        } else {
            mutateMatrixActivation(input, ActivationFunc.ReLU);
        }
    }

    public static float derivativeReLU(float input) {
        return input > 0 ? 1 : 0;
    }

    /**
     * This function mutates an input matrix in place by applying the
     * {@link <a href="https://en.wikipedia.org/wiki/Sigmoid_function">Sigmoid function</a>}
     * to each value in the matrix, which is mathematically expressed as:
     * <p><code>1</code> / <code>(1 + Math.exp(-x))</code>, for any real number <code>x</code>.
     * @param input a 'NArray' object
     */
    public static void Sigmoid(NArray input) {
        if (input.dimension == 1) {
            mutateVectorActivation(input, ActivationFunc.Sigmoid);
        } else {
            mutateMatrixActivation(input, ActivationFunc.Sigmoid);
        }
    }

    public static float derivativeSigmoid(float input) {
        return input * (1 - input);
    }

    /**
     * Uses the Softmax algorithm to compute outputs into a probability distribution
     * represented via floats, by mutating the input matrix in place.
     * @param input a 'NArray' object
     */
    public static void Softmax(NArray input) {
        if (input.dimension == 1) {
            softmaxVector(input);
        } else {
            softmaxMatrix(input);
        }
    }

    private static void softmaxVector(NArray vec) {
        int n = vec.getVecArray().length;
        float max = Float.NEGATIVE_INFINITY;

        // determine max value within array to avoid overflow
        for (int i = 0; i < n; i++) {
            float val = vec.get(i);
            if (val > max) {
                max = val;
            }
        }

        // subtract each elem by max while applying exponentiation and sum accumulation
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            float exp = (float) Math.exp(vec.get(i) - max);
            vec.set(i, exp);
            sum += exp;
        }

        // divide by sum for final softmax output
        for (int i = 0; i < n; i++) {
            vec.set(i, vec.get(i) / sum);
        }
    }

    private static void softmaxMatrix(NArray matrix) {
        int rows = matrix.shape[0];
        int cols = matrix.shape[1];

        for (int i = 0; i < rows; i++) {
            // determine max in row to avoid overflow
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) {
                float val = matrix.get(i, j);
                if (val > max) {
                    max = val;
                }
            }

            // subtract each elem by max while applying exponentiation and sum accumulation for this row
            float sum = 0f;
            for (int j = 0; j < cols; j++) {
                float exp = (float) Math.exp(matrix.get(i, j) - max);
                matrix.set(i, j, exp);
                sum += exp;
            }

            // divide by sum for final softmax output for this row
            for (int j = 0; j < cols; j++) {
                matrix.set(i, j, matrix.get(i, j) / sum);
            }
        }
    }

    private static void mutateVectorActivation(NArray vec, ActivationFunc func) {
        for (int i = 0; i < vec.getVecArray().length; i++) {
            float val = vec.get(i);
            switch (func) {
                case Sigmoid: {
                    vec.set(i, (float) (1 / (1 + Math.exp(-val))));
                }
                case ReLU: {
                    vec.set(i, Math.max(0, val));
                }
            }
        }
    }

    private static void mutateMatrixActivation(NArray matrix, ActivationFunc func) {
        float[][] array = matrix.getMatrixArray();

        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                float val = array[i][j];
                switch (func) {
                    case Sigmoid: {
                        matrix.set(i, j, (float) (1 / (1 + Math.exp(-val))));
                    }
                    case ReLU: {
                        matrix.set(i, j, Math.max(0, val));
                    }
                }
            }
        }
    }
}
