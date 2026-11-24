package org.sdws.mathematics;

public class Softmax implements Activation {
    @Override
    public NArray compute(NArray input) {
        float max = Float.NEGATIVE_INFINITY;
        float sum = 0f;

        if (input.dimension == 1) {
            float[] output = new float[input.getVecArray().length];

            // get max
            for (int i = 0; i < output.length; i++) {
                max = Math.max(input.get(i), max);
            }

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

            // get max
            for (int i = 0; i < output.length; i++) {
                for (int j = 0; j < output[i].length; j++) {
                    max = Math.max(input.get(i, j), max);
                }
            }

            // subtract elements by max and accumulate
            for (int i = 0; i < output.length; i++) {
                for (int j = 0; j < output[i].length; j++) {
                    float exp = (float) Math.exp(input.get(i, j) - max);
                    output[i][j] = exp;
                    sum += exp;
                }
            }

            // divide
            for (int i = 0; i < output.length; i++) {
                for (int j = 0; j < output[i].length; j++) {
                    output[i][j] /= sum;
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
