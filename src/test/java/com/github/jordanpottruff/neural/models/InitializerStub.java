package com.github.jordanpottruff.neural.models;

import com.github.jordanpottruff.neural.initializers.Initializer;

import java.util.List;

// Stub used for initializing weights in a predictable manner.
public class InitializerStub implements Initializer {

    private final List<Double> values;
    private int nextIndex = 0;

    // Creates the stub which will produce the given values when a method is called.
    public InitializerStub(List<Double> values) {
        this.values = values;
    }


    public double getWeight(int layer) {
        double value = values.get(nextIndex);
        incrementIndex();
        return value;
    }

    public void incrementIndex() {
        nextIndex = (nextIndex + 1) % values.size();
    }
}
