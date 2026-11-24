package org.sdws.mathematics;

public class ReLU implements Activation {
    @Override
    public NArray compute(NArray input) {
        if (input.dimension == 1) {
            float[] output = new float[input.getVecArray().length];

            for (int i = 0; i < input.getVecArray().length; i++) {
                output[i] = Math.max(0, input.get(i));
            }

            return NArray.FromElements(output);
        } else {
            float[][] array = new float[input.shape[0]][input.shape[1]];

            for (int i = 0; i < array.length; i++) {
                for (int j = 0; j < array[i].length; j++) {
                    array[i][j] = Math.max(0, input.get(i, j));
                }
            }

            return NArray.FromNested(array);
        }
    }

    @Override
    public float computeDerivative(float input) {
        return input > 0 ? 1 : 0;
    }
}
