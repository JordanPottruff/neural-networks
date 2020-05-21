package com.github.jordanpottruff.neural.activations;

import com.github.jordanpottruff.jgml.VecN;

/**
 * Logistic function implementation.
 */
public class Logistic implements ActivationFunc {

    /**
     * Applies the standard logistic function to the vector.
     * @param vec the input vector.
     * @return a new vector formed by applying the logistic function to the input.
     */
    @Override
    public VecN applyFunc(VecN vec) {
        double[] values = vec.toArray();
        for (int i = 0; i < values.length; i++) {
            values[i] = 1.0 / (1.0 + Math.exp(-values[i]));
        }
        return new VecN(values);
    }

    /**
     * Applies the derivative of the standard logistic function to the vector.
     * @param vec the input vector.
     * @return a new vector formed by applying the derivative of the logistic function to the input.
     */
    public VecN applyPrime(VecN vec) {
        double[] values = vec.toArray();
        for (int i = 0; i < values.length; i++) {
            values[i] = values[i] * (1 - values[i]);
        }
        return new VecN(values);
    }
}
