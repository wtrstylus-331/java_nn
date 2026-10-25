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
     * Normalize output elements into 'probabilities' via Gaussian distribution.
     */
    Softmax,

    /**
     * No activation function.
     */
    None,
}
