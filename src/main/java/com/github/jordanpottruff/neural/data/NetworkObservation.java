package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.jgml.VecN;

/**
 * Defines an observation to be used in a neural network. That is, data that associates a handful of attribute values to
 * an observed classification.
 */
public interface NetworkObservation {

    /**
     * Returns the attributes of the observation.
     *
     * @return observations as a vector.
     */
    VecN getAttributes();

    /**
     * Returns the classification of the observation.
     *
     * @return name of observation's class.
     */
    String getClassification();
}