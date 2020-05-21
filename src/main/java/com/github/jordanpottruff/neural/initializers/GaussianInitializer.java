package com.github.jordanpottruff.neural.initializers;

import java.util.Random;

/**
 * Initializer that uses the gaussian distribution.
 */
public class GaussianInitializer implements Initializer {

    private final Random random = new Random();
    private final double mean;
    private final double stdDev;

    /**
     * Generates weights according to the standard gaussian distribution.
     */
    public GaussianInitializer() {
        this(0.0, 1.0);
    }

    /**
     * Generates weights according to a gaussian distribution centered on zero with the specified standard deviation.
     * @param stdDev the standard deviation of the distribution.
     */
    public GaussianInitializer(double stdDev) {
        this(0.0, stdDev);
    }

    /**
     * Generates weights according to a gaussian distribution centered on the specified mean with the specified standard
     * deviation.
     * @param mean the mean of the distribution.
     * @param stdDev the standard deviation of the distribution.
     */
    public GaussianInitializer(double mean, double stdDev) {
        this.mean = mean;
        this.stdDev = stdDev;
    }

    /**
     * @inheritDoc
     */
    @Override
    public double getWeight(int prevLayerSize) {
        return (random.nextGaussian()*stdDev)+mean;
    }
}
