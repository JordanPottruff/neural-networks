package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.jgml.MatMN;
import com.github.jordanpottruff.jgml.VecN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Creates basic neural networks that learn with back propagation.
 */
public class BackPropNetwork implements Network {

    private final List<Integer> layerSizes;
    private final String[] classes;
    private final double learningRate;
    private final double momentum;
    private final List<MatMN> weights;
    private final List<VecN> biases;

    /**
     * Creates a new back propagation neural network.
     *
     * @param inputSize the number of nodes in the input layer.
     * @param hiddenSizes the number of nodes in each hidden layer. Each value corresponds to successive layer sizes.
     * @param classes the classifications an observation can receive.
     * @param learningRate the learning rate for the back propagation algorithm.
     * @param momentum the momentum for the back propagation algorithm.
     */
    public BackPropNetwork(int inputSize, int[] hiddenSizes, String[] classes, double learningRate, double momentum) {
        this.layerSizes = createLayerSizes(inputSize, hiddenSizes, classes);
        this.classes = classes;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.weights = this.generateRandomWeights(this.layerSizes);
        this.biases = this.generateRandomBiases(this.layerSizes);
    }

    private List<Integer> createLayerSizes(int inputSize, int[] hiddenSizes, String[] classes) {
        List<Integer> layerSizes = new ArrayList<>();
        layerSizes.add(inputSize);
        layerSizes.addAll(Arrays.stream(hiddenSizes).boxed().collect(Collectors.toList()));
        layerSizes.add(classes.length);
        return layerSizes;
    }

    // Returns a list of random weight matrices that conform to the specifications of the layer sizes.
    private List<MatMN> generateRandomWeights(List<Integer> layerSizes) {
        List<MatMN> randomWeights = new ArrayList<>();
        for(int i=1; i<layerSizes.size(); i++) {
            int rows = layerSizes.get(i);
            int cols = layerSizes.get(i-1);
            randomWeights.add(randomMatrix(rows, cols));
        }
        return randomWeights;
    }

    // Returns a single random matrix of size rows x cols.
    private MatMN randomMatrix(int rows, int cols) {
        Random random = new Random();
        double[][] values = new double[cols][rows];
        for(int c=0; c<cols; c++) {
            for(int r=0; r<rows; r++) {
                values[c][r] = random.nextGaussian();
            }
        }
        return new MatMN(values);
    }

    // Returns a list of random bias vectors that conform to the specifications of the layer sizes.
    private List<VecN> generateRandomBiases(List<Integer> layerSizes) {
        List<VecN> randomBiases = new ArrayList<>();
        // No bias is added to the input layer, so we begin at i=1.
        for(int i=1; i<layerSizes.size(); i++) {
            randomBiases.add(randomVector(layerSizes.get(i)));
        }
        return randomBiases;
    }

    // Returns a single random vector of the given dimension.
    private VecN randomVector(int dimension) {
        Random random = new Random();
        double[] values = new double[dimension];
        for(int i=0; i<dimension; i++) {
            values[i] = random.nextGaussian();
        }
        return new VecN(values);
    }

    /**
     * @inheritDoc
     */
    @Override
    public void train(DataSet trainingSet, double validationProportion) {

    }

    /**
     * @inheritDoc
     */
    @Override
    public BackPropResult test(DataSet testingSet) {
        return null;
    }

    /**
     * @inheritDoc
     */
    @Override
    public String classify(VecN attributes) {
        return null;
    }

    /**
     * Returns a string representation of the network.
     *
     * @return the string representation of the network.
     */
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("Weights: ").append("\n");
        for(int i=0; i<weights.size(); i++) {
            result.append("Layer ").append(i).append("\n");
            result.append(weights.get(i).toString()).append("\n");
        }
        result.append("Biases: ").append("\n");
        for(int i=0; i<biases.size(); i++) {
            result.append("Layer ").append(i+1).append("\n");
            result.append(biases.get(i).toString()).append("\n");
        }
        return result.toString();
    }

    /**
     * The results of testing a back propagation neural network.
     */
    public static class BackPropResult implements Network.TestResult {

        private final double accuracy;
        private final double error;
        private final List<Observation> correct;
        private final List<Observation> incorrect;

        // Compiles the data to create a result of a test on a back prop network.
        private BackPropResult(double accuracy, double error, List<Observation> correct, List<Observation> incorrect) {
            this.accuracy = accuracy;
            this.error = error;
            this.correct = correct;
            this.incorrect = incorrect;
        }

        /**
         * @inheritDoc
         */
        @Override
        public double getAccuracy() {
            return this.accuracy;
        }

        /**
         * @inheritDoc
         */
        @Override
        public double getError() {
            return this.error;
        }

        /**
         * @inheritDoc
         */
        @Override
        public List<Observation> correct() {
            return new ArrayList<>(this.correct);
        }

        /**
         * @inheritDoc
         */
        @Override
        public List<Observation> incorrect() {
            return new ArrayList<>(this.incorrect);
        }
    }

}
