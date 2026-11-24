package org.sdws.mathematics;

public class Sigmoid implements Activation {
    @Override
    public NArray compute(NArray input) {
        if (input.dimension == 1) {
            float[] output = new float[input.getVecArray().length];

            for (int i = 0; i < input.getVecArray().length; i++) {
                output[i] = (float) (1 / (1 + Math.exp(-input.get(i))));
            }

            return NArray.FromElements(output);
        } else {
            float[][] array = new float[input.shape[0]][input.shape[1]];

            for (int i = 0; i < array.length; i++) {
                for (int j = 0; j < array[i].length; j++) {
                    array[i][j] = (float) (1 / (1 + Math.exp(-input.get(i, j))));
                }
            }

            return NArray.FromNested(array);
        }
    }

    @Override
    public float computeDerivative(float input) {
        return input * (1 - input);
    }
}
