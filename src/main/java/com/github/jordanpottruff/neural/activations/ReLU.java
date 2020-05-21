package com.github.jordanpottruff.neural.activations;

import com.github.jordanpottruff.jgml.VecN;

import java.util.function.Function;

/**
 * Rectifier function implementation.
 */
public class ReLU implements Function<VecN, VecN> {

    /**
     * Applies the rectified activation function to the vector.
     * @param vec the input vector.
     * @return a new vector formed by applying the rectified function to the input.
     */
    @Override
    public VecN apply(VecN vec) {
        double[] values = vec.toArray();
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.max(0.0, values[i]);
        }
        return new VecN(values);
    }

    /**
     * The derivative of the rectifier function.
     */
    public static class Prime implements Function<VecN, VecN> {

        /**
         * Applies the derivative of the rectifier function to the vector.
         * @param vec the input vector.
         * @return a new vector formed by applying the rectifier function's derivative to the input.
         */
        @Override
        public VecN apply(VecN vec) {
            double[] values = vec.toArray();
            for (int i = 0; i < values.length; i++) {
                values[i] = values[i] > 0 ? 1 : 0;

            }
            return new VecN(values);
        }
    }
}
