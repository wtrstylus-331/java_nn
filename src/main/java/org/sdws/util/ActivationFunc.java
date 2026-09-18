package org.sdws.util;

/**
 * Enumerator class that holds various types of activation functions.
 */
public enum ActivationFunc {
    /**
     * Rectified linear unit, essentially {@code max(0,x)}.
     */
    ReLU,

    /**
     * Sigmoid function.
     */
    Sigmoid,

    /**
     * Use the step function, mapping inputs to either {@code 0} or
     * {@code 1} based on sign.
     */
    Step,

    /**
     * No activation function.
     */
    None,

    /**
     * Normalize output elements into 'probabilities' via Gaussian distribution.
     */
    Softmax
}
