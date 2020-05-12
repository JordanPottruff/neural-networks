package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.jgml.VecN;

import java.util.List;

/**
 * Defines neural network behavior, including training and testing.
 */
public interface Network {

    /**
     * Trains the neural network on the given data set.
     *
     * @param trainingSet the DataSet to train the network on.
     */
    void train(DataSet trainingSet);

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
