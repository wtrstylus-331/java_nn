package org.sdws.mathematics;

public class Softmax implements Activation {
    @Override
    public NArray compute(NArray input) {
        if (input.dimension == 1) {
            float[] output = new float[input.getVecArray().length];
            float max = Float.NEGATIVE_INFINITY;

            // get max
            for (int i = 0; i < output.length; i++) {
                max = Math.max(input.get(i), max);
            }

            float sum = 0f;
            // subtract elements by max and accumulate
            for (int i = 0; i < output.length; i++) {
                float exp = (float) Math.exp(input.get(i) - max);
                output[i] = exp;
                sum += exp;
            }

            // divide
            for (int i = 0; i < output.length; i++) {
                output[i] /= sum;
            }

            return NArray.FromElements(output);
        } else {
            float[][] output = new float[input.shape[0]][input.shape[1]];
            int rows = input.shape[0];
            int cols = input.shape[1];

            for (int row = 0; row < rows; row++) {
                float max = Float.NEGATIVE_INFINITY;

                // get max
                for (int col = 0; col < cols; col++) {
                    max = Math.max(input.get(row, col), max);
                }

                float sum = 0f;
                // subtract elements by max and accumulate for this row
                for (int col = 0; col < cols; col++) {
                    float exp = (float) Math.exp(input.get(row, col) - max);
                    output[row][col] = exp;
                    sum += exp;
                }

                // divide elements by sum for this row
                for (int col = 0; col < cols; col++) {
                    output[row][col] /= sum;
                }
            }

            return NArray.FromNested(output);
        }
    }

    @Override
    public float computeDerivative(float input) {
        //todo: implement matrix derivative
        throw new UnsupportedOperationException("softmax derivative not implemented yet");
    }
}
