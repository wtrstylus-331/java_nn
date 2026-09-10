package org.sdws.mathematics;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Activation {
    /**
     * ReLU stands for {@link <a href="https://en.wikipedia.org/wiki/Rectified_linear_unit">'Rectified Linear Unit'</a>},
     * which can also be expressed as:
     * <p><code>Math.max(0, x)</code> for any real number <code>x</code>. <p>This function mutates
     * an input matrix in place by applying the ReLU function to each value in the matrix.
     * @param input a 'NNArray' object
     */
    public static void ReLU(NNArray input) {
        recurseWithFunction(input.nnarray(), ActivationFunc.ReLU);
    }

    public static void ReLU(NArray input) {
        recurseAFunction(input.innerArray(), ActivationFunc.ReLU);
    }

    /**
     * This function mutates an input matrix in place by applying the
     * {@link <a href="https://en.wikipedia.org/wiki/Sigmoid_function">Sigmoid function</a>}
     * to each value in the matrix, which is mathematically expressed as:
     * <p><code>1</code> / <code>(1 + Math.exp(-x))</code>, for any real number <code>x</code>.
     * @param input a 'NNArray' object
     */
    public static void Sigmoid(NNArray input) {
        recurseWithFunction(input.nnarray(), ActivationFunc.Sigmoid);
    }
    public static void Sigmoid(NArray input) {
        recurseAFunction(input.innerArray(), ActivationFunc.Sigmoid);
    }

    /**
     * This function mutates an input matrix in place
     * to each value in the matrix while applying the following function:
     * <p> f(x):
     * <p><code>0</code> if x <= 0
     * <p><code>1</code> if x > 0
     * @param input a 'NNArray' object
     */
    public static void Step(NNArray input) {
        recurseWithFunction(input.nnarray(), ActivationFunc.Step);
    }
    public static void Step(NArray input) {
        recurseAFunction(input.innerArray(), ActivationFunc.Step);
    }

    /**
     * This function mutates an input matrix in place by applying
     * the hyperbolic tangent function to each input, resulting in
     * outputs ranging from {@code -1} to {@code}.
     */
    public static void Tanh(NNArray input) {
        recurseWithFunction(input.nnarray(), ActivationFunc.Tanh);
    }
    public static void Tanh(NArray input) {
        recurseAFunction(input.innerArray(), ActivationFunc.Tanh);
    }

    /**
     * Uses the Softmax algorithm to compute outputs into a probability distribution
     * represented via floats, by mutating the input matrix in place.
     * @param input a 'NNArray' object
     */
    public static void Softmax(NNArray input) {
        recurseSoftmax(input.nnarray());
    }
    public static void Softmax(NArray input) {
        recurseSoftmaxAlgo(input.innerArray());
    }

    private static void recurseSoftmaxAlgo(Object input) {
        if (input instanceof Number) {
            return;
        }

        ArrayList<?> list = (ArrayList<?>) input;
        Class<?> type = list.get(0).getClass();
        Number max;

        // determine max value within array to avoid overflow
        if (type.equals(Float.class)) {
            max = Collections.max((ArrayList<Float>) list);
        } else if (type.equals(Double.class)) {
            max = Collections.max((ArrayList<Double>) list);
        } else if (type.equals(Integer.class)) {
            max = Collections.max((ArrayList<Integer>) list);
        } else if (type.equals(Long.class)) {
            max = Collections.max((ArrayList<Long>) list);
        } else if (type.equals(Short.class)) {
            max = Collections.max((ArrayList<Short>) list);
        } else if (type.equals(Byte.class)) {
            max = Collections.max((ArrayList<Byte>) list);
        }
    }

    private static void recurseSoftmax(Object input) {
        if (input instanceof Float) {
            return;
        }

        List<?> list = (List<?>) input;
        float sum = 0f;
        float max = Float.NEGATIVE_INFINITY;

        // determine max value within array to avoid overflow
        for (int i = 0; i < list.size(); i++) {
            Object element = list.get(i);

            if (element instanceof Float f) {
                if (f > max) {
                    max = f;
                }
            } else {
                recurseSoftmax(element);
            }
        }

        // subtract values by max and exponentiate for softmax algo
        for (int i = 0; i < list.size(); i++) {
            Object element = list.get(i);

            if (element instanceof Float) {
                ((List<Float>) list).set(i, ((List<Float>) list).get(i) - max);

                float exponentiated = (float) Math.exp(((List<Float>) list).get(i));
                ((List<Float>) list).set(i, exponentiated);
                sum += exponentiated;
            } else {
                recurseSoftmax(element);
            }
        }

        // divide each value by accumulated sum for softmax algo
        for (int i = 0; i < list.size(); i++) {
            Object element = list.get(i);

            if (element instanceof Float f) {
                float finalVal = f / sum;
                ((List<Float>) list).set(i, Math.round(finalVal * 10000) / 10000f);
            } else {
                recurseSoftmax(element);
            }
        }
    }

    private static void recurseAFunction(Object input, ActivationFunc function) {
        if (input instanceof Number) {
            return;
        }

        ArrayList<?> list = (ArrayList<?>) input;

        for (int i = 0; i < list.size(); i++) {
            Object element = list.get(i);
            if (element instanceof Number) {
                switch (element.getClass().getSimpleName()) {
                    case "Float" -> mutateArrayValues(list, element, Float.class, function);
                    case "Double" -> mutateArrayValues(list, element, Double.class, function);
                    case "Integer" -> mutateArrayValues(list, element, Integer.class, function);
                    case "Long" -> mutateArrayValues(list, element, Long.class, function);
                    case "Short" -> mutateArrayValues(list, element, Short.class, function);
                    case "Byte" -> mutateArrayValues(list, element, Byte.class, function);
                }
            } else {
                recurseAFunction(list.get(i), function);
            }
        }
    }

    private static void mutateArrayValues(ArrayList<?> array, Object value, Class<?> type, ActivationFunc function) {
        int i = array.indexOf(value); // store index due to being out of loop scope
        switch (function) {
            case ReLU -> {
                if (type.equals(Float.class)) {
                    ((ArrayList<Float>) array).set(i, Math.max(0, (Float) value));
                } else if (type.equals(Double.class)) {
                    ((ArrayList<Double>) array).set(i, Math.max(0, (Double) value));
                } else if (type.equals(Integer.class)) {
                    ((ArrayList<Integer>) array).set(i, Math.max(0, (Integer) value));
                } else if (type.equals(Long.class)) {
                    ((ArrayList<Long>) array).set(i, Math.max(0, (Long) value));
                } else if (type.equals(Short.class)) {
                    ((ArrayList<Short>) array).set(i, (short) Math.max(0, (Short) value));
                } else if (type.equals(Byte.class)) {
                    ((ArrayList<Byte>) array).set(i, (byte) Math.max(0, (Byte) value));
                }
            }
            case Sigmoid -> {
                if (type.equals(Float.class)) {
                    ((ArrayList<Float>) array).set(i, (float) (1 / (1 + Math.exp(-(Float) value))));
                } else if (type.equals(Double.class)) {
                    ((ArrayList<Double>) array).set(i, 1 / (1 + Math.exp(-(Double) value)));
                } else if (type.equals(Integer.class)) {
                    ((ArrayList<Integer>) array).set(i, (int) (1 / (1 + Math.exp(-(Integer) value))));
                } else if (type.equals(Long.class)) {
                    ((ArrayList<Long>) array).set(i, (long) (1 / (1 + Math.exp(-(Long) value))));
                } else if (type.equals(Short.class)) {
                    ((ArrayList<Short>) array).set(i, (short) (1 / (1 + Math.exp(-(Short) value))));
                } else if (type.equals(Byte.class)) {
                    ((ArrayList<Byte>) array).set(i, (byte) (1 / (1 + Math.exp(-(Byte) value))));
                }
            }
            case Step -> {
                if (type.equals(Float.class)) {
                    ((ArrayList<Float>) array).set(i, (Float) value < 0 ? 0 : 1f);
                } else if (type.equals(Double.class)) {
                    ((ArrayList<Double>) array).set(i, (Double) value < 0 ? 0 : 1d);
                } else if (type.equals(Integer.class)) {
                    ((ArrayList<Integer>) array).set(i, (int) value < 0 ? 0 : 1);
                } else if (type.equals(Long.class)) {
                    ((ArrayList<Long>) array).set(i, (long) value < 0 ? 0 : 1L);
                } else if (type.equals(Short.class)) {
                    ((ArrayList<Short>) array).set(i, (short) ((short) value < 0 ? 0 : 1));
                } else if (type.equals(Byte.class)) {
                    ((ArrayList<Byte>) array).set(i, (byte) ((byte) value < 0 ? 0 : 1));
                }
            }
        }
    }

    private static void recurseWithFunction(Object input, ActivationFunc activationFunction) {
        if (input instanceof Float) {
            return;
        }

        List<?> list = (List<?>) input;

        for (int i = 0; i < list.size(); i++) {
            Object element = list.get(i);

            if (element instanceof Float f) {
                switch (activationFunction) {
                    case ReLU -> {
                        float val = Math.round(((List<Float>) list).get(i) * 10000) / 10000f;
                        ((List<Float>) list).set(i, Math.max(0, val));
                    }
                    case Sigmoid -> {
                        float val = ((List<Float>) list).get(i);
                        float output = (float) (1 / (1 + Math.exp(-val)));
                        ((List<Float>) list).set(i, (float)Math.round(output * 10000) / 10000f);
                    }
                    case Step -> {
                        if (f < 0f) {
                            ((List<Float>) list).set(i, 0f);
                        } else {
                            ((List<Float>) list).set(i, 1f);
                        }
                    }
                    case Tanh -> {
                        float val = Math.round(((List<Float>) list).get(i) * 10000) / 10000f;
                        float output = (float) ((Math.exp(val) - Math.exp(-val)) / (Math.exp(val) + Math.exp(-val)));
                        ((List<Float>) list).set(i, output);
                    }
                }
            } else {
                recurseWithFunction(element, activationFunction);
            }
        }
    }
}
