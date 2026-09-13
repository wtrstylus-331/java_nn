package org.sdws.util;

/**
 * Enumerator class that holds various types of activation functions.
 */
public enum ActivationFunc {
    /**
     * Rectified linear unit.
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
     * Use the hyperbolic tangent function to map inputs between a range
     * of {@code -1} to {@code 1}.
     */
    Tanh,

    /**
     * No activation function.
     */
    None,

    /**
     * Normalize vector elements via Gaussian distribution.
     */
    Softmax
}
