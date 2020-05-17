package com.github.jordanpottruff.neural.data;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Data sets are composed of a collection of observations and additional information about the classification of these
 * observations.
 */
public class DataSet implements NetworkDataSet {

    private final ArrayList<Observation> observations;
    private final String[] classes;

    /**
     * Creates a new data set from a collection of observations.
     *
     * @param observations the observations.
     * @param classes      the classes that the observations can belong to.
     */
    public DataSet(Collection<Observation> observations, String[] classes) {
        this.observations = new ArrayList<>(observations);
        this.classes = classes;
    }

    /**
     * @inheritDoc
     */
    @Override
    public int size() {
        return observations.size();
    }

    /**
     * @inheritDoc
     */
    @Override
    public String[] getClasses() {
        String[] classesCopy = new String[this.classes.length];
        System.arraycopy(this.classes, 0, classesCopy, 0, this.classes.length);
        return classesCopy;
    }

    /**
     * @inheritDoc
     */
    @Override
    public Observation getObservation(int i) {
        return observations.get(i);
    }

    /**
     * @inheritDoc
     */
    @Override
    public List<Observation> getAllObservations() {
        return new ArrayList<>(observations);
    }
}
