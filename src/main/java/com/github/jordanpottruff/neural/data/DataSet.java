package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.neural.common.Pair;

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

    /**
     * @inheritDoc
     */
    @Override
    public Pair<DataSet, DataSet> split(double percentage) {
        int cutoff = (int) Math.ceil(size() * percentage - 1);
        List<Observation> left = new ArrayList<>();
        List<Observation> right = new ArrayList<>();

        for(int i=0; i<size(); i++) {
            if(i <= cutoff) {
                left.add(getObservation(i));
            } else {
                right.add(getObservation(i));
            }
        }
        return new Pair<>(new DataSet(left, classes), new DataSet(right, classes));
    }
}
