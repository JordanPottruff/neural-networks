package com.github.jordanpottruff.neural.initializers;

import java.util.Random;

/**
 * Initializer for rectifier functions defined by He, et al. (2015).
 */
public class HeInitializer implements Initializer{

    private final Random random = new Random();

    /**
     * @inheritDoc
     */
    @Override
    public double getWeight(int prevLayerSize) {
        return random.nextGaussian() * Math.sqrt(2.0 / prevLayerSize);
    }
}
