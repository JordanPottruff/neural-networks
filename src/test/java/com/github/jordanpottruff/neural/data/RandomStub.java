package com.github.jordanpottruff.neural.data;

import java.util.List;
import java.util.Random;

// Stub of Random class that produces predetermined values.
public class RandomStub extends Random {

    private final List<Double> values;
    private int nextIndex = 0;

    // Creates the stub which will produce the given values when a method is called.
    public RandomStub(List<Double> values) {
        this.values = values;
    }

    @Override
    public double nextGaussian() {
        double value = values.get(nextIndex);
        incrementIndex();
        return value;
    }

    public void incrementIndex() {
        nextIndex = (nextIndex + 1) % values.size();
    }
}
