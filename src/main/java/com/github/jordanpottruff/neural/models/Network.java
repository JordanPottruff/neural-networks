package com.github.jordanpottruff.neural.models;

import com.github.jordanpottruff.jgml.VecN;
import com.github.jordanpottruff.neural.data.DataSet;
import com.github.jordanpottruff.neural.data.Observation;

import java.util.List;

/**
 * Defines neural network behavior, including training and testing.
 */
public interface Network {

    /**
     * Trains the neural network on the given training data set by separating the data into a series of mini-batches
     * and performing gradient descent on each. This can be thought of as a single training epoch.
     *
     * @param trainingSet   the DataSet to train the network on.
     * @param miniBatchSize the size of the mini-batches to be used in gradient descent.
     * @param learningRate  the learning rate for the epoch.
     */
    void train(DataSet trainingSet, int miniBatchSize, double learningRate);

    /**
     * Tests the neural network on the given data set.
     *
     * @param testingSet the DataSet to test the network on.
     * @return the results of the test.
     */
    TestResult test(DataSet testingSet);

    /**
     * Classifies a given set of attributes according to the current network weights.
     *
     * @param attributes a list of attribute values.
     * @return the classification of the attribute values according to the network.
     */
    String classify(VecN attributes);

    /**
     * Defines the result of testing a neural network.
     */
    interface TestResult {

        /**
         * Returns the accuracy of the neural network's classification of the test data.
         *
         * @return the accuracy over the entire data set.
         */
        double getAccuracy();

        /**
         * Returns the error of the neural network's classification of the test data.
         *
         * @return the error over the entire data set.
         */
        double getError();

        /**
         * Returns a list of the observations that were correctly classified.
         *
         * @return the correctly classified observations.
         */
        List<Observation> correct();

        /**
         * Returns a list of the observations that were incorrectly classified.
         *
         * @return the incorrectly classified observations.
         */
        List<Observation> incorrect();

    }
}
