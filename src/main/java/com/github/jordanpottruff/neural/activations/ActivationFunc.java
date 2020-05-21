package com.github.jordanpottruff.neural.activations;

import com.github.jordanpottruff.jgml.VecN;

/**
 * Defines an activation function.
 */
public interface ActivationFunc {

    /**
     * Applies the activation function to the input.
     * @param input the input to the function.
     * @return the output of the activation function.
     */
    VecN applyFunc(VecN input);

    /**
     * Applies the derivative of the activation function to the input.
     * @param input the input to the function's derivative.
     * @return the output of the derivative function.
     */
    VecN applyPrime(VecN input);
}
