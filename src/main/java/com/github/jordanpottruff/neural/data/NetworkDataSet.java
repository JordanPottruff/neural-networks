package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.neural.common.Pair;

import java.util.List;

/**
 * Defines a data set to be used in a neural network. A data set is composed of a series of observations.
 */
public interface NetworkDataSet {

    /**
     * Returns the number of observations in the data set.
     *
     * @return the number of observations in the data set.
     */
    int size();

    String[] getClasses();

    /**
     * Returns the ith observation in the data set.
     *
     * @param i the index of the observation to be returned.
     * @return the observation at index i.
     */
    NetworkObservation getObservation(int i);

    /**
     * Returns a list of all the observations in the data set.
     *
     * @return the list of observations.
     */
    List<? extends NetworkObservation> getAllObservations();

    /**
     * Splits the data in the current data set into two new data sets.
     * @param percentage the percentage of data in the first returned data set (key of returned pair).
     * @return a pair of data sets.
     */
    Pair<DataSet, DataSet> split(double percentage);

    /**
     * Shuffles the data.
     */
    void shuffle();

}
