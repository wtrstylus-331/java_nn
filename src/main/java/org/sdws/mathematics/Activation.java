package org.sdws.mathematics;

public interface Activation {
    NArray compute(NArray input);
    float computeDerivative(float input);
}
