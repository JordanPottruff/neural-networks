package com.github.jordanpottruff.neural.initializers;

/**
 * Defines an initialization scheme for network weights.
 */
public interface Initializer {

    /**
     * Returns a newly generated weight.
     * @param prevLayerSize the number of nodes in the previous layer.
     * @return the new weight.
     */
    double getWeight(int prevLayerSize);
}
