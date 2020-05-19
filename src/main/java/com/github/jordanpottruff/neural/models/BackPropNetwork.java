package com.github.jordanpottruff.neural.models;

import com.github.jordanpottruff.jgml.MatMN;
import com.github.jordanpottruff.jgml.VecN;
import com.github.jordanpottruff.neural.common.Pair;
import com.github.jordanpottruff.neural.data.DataSet;
import com.github.jordanpottruff.neural.data.Observation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Creates basic neural networks that learn with back propagation.
 */
public class BackPropNetwork implements Network {

    final List<Integer> layerSizes;
    final String[] classes;
    final double learningRate;
    final double momentum;
    final List<MatMN> weights;
    final List<VecN> biases;
    final Random rand;

    /**
     * Creates a new back propagation neural network.
     *
     * @param inputSize    the number of nodes in the input layer.
     * @param hiddenSizes  the number of nodes in each hidden layer. Each value corresponds to successive layer sizes.
     * @param classes      the classifications an observation can receive.
     * @param learningRate the learning rate for the back propagation algorithm.
     * @param momentum     the momentum for the back propagation algorithm.
     */
    public BackPropNetwork(int inputSize, int[] hiddenSizes, String[] classes, double learningRate, double momentum) {
        this(new Random(), inputSize, hiddenSizes, classes, learningRate, momentum);
    }

    // Additional constructor that can be used for testing.
    BackPropNetwork(Random rand, int inputSize, int[] hiddenSizes, String[] classes, double learningRate, double momentum) {
        this.rand = rand;
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
        for (int i = 1; i < layerSizes.size(); i++) {
            int rows = layerSizes.get(i);
            int cols = layerSizes.get(i - 1);
            randomWeights.add(randomMatrix(rows, cols));
        }
        return randomWeights;
    }

    // Returns a single random matrix of size rows x cols.
    private MatMN randomMatrix(int rows, int cols) {
        double[][] values = new double[cols][rows];
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                values[c][r] = rand.nextGaussian();
            }
        }
        return new MatMN(values);
    }

    // Returns a list of random bias vectors that conform to the specifications of the layer sizes.
    private List<VecN> generateRandomBiases(List<Integer> layerSizes) {
        List<VecN> randomBiases = new ArrayList<>();
        // No bias is added to the input layer, so we begin at i=1.
        for (int i = 1; i < layerSizes.size(); i++) {
            randomBiases.add(randomVector(layerSizes.get(i)));
        }
        return randomBiases;
    }

    // Returns a single random vector of the given dimension.
    private VecN randomVector(int dimension) {
        double[] values = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            values[i] = rand.nextGaussian();
        }
        return new VecN(values);
    }

    /**
     * @inheritDoc
     */
    @Override
    public void train(DataSet trainingSet, int miniBatchSize, double validationProportion) {
        // Split out validation data from the training data.
        Pair<DataSet, DataSet> data = trainingSet.split(1 - validationProportion);
        DataSet training = data.getKey();
        DataSet validation = data.getValue();
    }

    public Pair<MatMN[], VecN[]> calculateGradient(Observation obs) {
        List<VecN> activations = getActivations(obs.getAttributes());
        VecN expectedOutput = getExpectedOutput(obs.getClassification());

        // Two gradients: one for weights, one for biases.
        MatMN[] weightGradient = new MatMN[weights.size()];
        VecN[] biasGradient = new VecN[biases.size()];

        // Calculate gradient of output layer.
        int outputLayerIndex = activations.size() - 1;
        VecN outputActivation = activations.get(outputLayerIndex);
        VecN logisticPrime = Util.applyLogisticPrime(outputActivation);
        VecN delta = Util.componentWiseMultiply(expectedOutput.subtract(outputActivation), logisticPrime).invert();

        weightGradient[outputLayerIndex - 1] = Util.outerProduct(delta, activations.get(outputLayerIndex - 1));
        biasGradient[outputLayerIndex - 1] = delta;

        // Calculate gradient of hidden layer(s).
        for (int hiddenLayerIndex = outputLayerIndex - 1; hiddenLayerIndex > 0; hiddenLayerIndex -= 1) {
            VecN hiddenActivation = activations.get(hiddenLayerIndex);
            logisticPrime = Util.applyLogisticPrime(hiddenActivation);
            VecN deltaTimesWeight = Util.transposeMultiply(delta, weights.get(hiddenLayerIndex));
            delta = Util.componentWiseMultiply(logisticPrime, deltaTimesWeight);

            weightGradient[hiddenLayerIndex - 1] = Util.outerProduct(delta, activations.get(hiddenLayerIndex - 1));
            biasGradient[hiddenLayerIndex - 1] = delta;
        }
        return new Pair<>(weightGradient, biasGradient);
    }

    // Return the activations of each layer, excluding the input layer.
    List<VecN> getActivations(VecN attributes) {
        List<VecN> activations = new ArrayList<>();
        activations.add(attributes);

        for (int layer = 0; layer < weights.size(); layer++) {
            VecN input = weights.get(layer).multiply(activations.get(layer)).add(biases.get(layer));
            activations.add(Util.applyLogistic(input));
        }
        return activations;
    }

    // Return a vector describing the expected output for the given classification.
    VecN getExpectedOutput(String classification) {
        double[] expected = new double[classes.length];
        for (int i = 0; i < classes.length; i++) {
            // If the observation's class corresponds to the class at the current index, set the index in the returned
            // vector equal to 1.
            if (classification.equals(classes[i])) {
                expected[i] = 1.0;
                break;
            }
        }
        return new VecN(expected);
    }

    /**
     * @inheritDoc
     */
    @Override
    public Result test(DataSet testingSet) {
        // Declare metadata about the test.
        double error = 0;
        int correctCount = 0;
        List<Observation> correct = new ArrayList<>();
        List<Observation> incorrect = new ArrayList<>();
        // Iterate across all observations in the testing set.
        for(Observation obs: testingSet.getAllObservations()) {
            // Get activations for the observation and classify according to the current network.
            List<VecN> activations = getActivations(obs.getAttributes());
            String classification = classify(obs.getAttributes(), activations);
            // Evaluate accuracy.
            if(classification.equals(obs.getClassification())) {
                correctCount++;
                correct.add(obs);
            } else {
                incorrect.add(obs);
            }
            // Evaluate error.
            VecN expectedOutput = getExpectedOutput(obs.getClassification());
            VecN actualOutput = activations.get(activations.size()-1);
            error += getError(expectedOutput, actualOutput);
        }
        // Calculate and encapsulate final testing metadata into a result.
        double accuracy = (double) correctCount / testingSet.size();
        double avgError = error / testingSet.size();
        return new Result(accuracy, avgError, correct, incorrect);
    }

    private double getError(VecN expected, VecN actual) {
        VecN diff = expected.subtract(actual);
        return 0.5*diff.dot(diff);
    }

    /**
     * @inheritDoc
     */
    @Override
    public String classify(VecN attributes) {
        return classify(attributes, getActivations(attributes));
    }

    // Computes class based on given activations; allows for caching of activation calculations.
    private String classify(VecN attributes, List<VecN> activations) {
        VecN output = getActivations(attributes).get(layerSizes.size()-1);
        int maxIndex = 0;
        for(int i=1; i<classes.length; i++) {
            if(output.get(i) > output.get(maxIndex)) {
                maxIndex = i;
            }
        }
        return classes[maxIndex];
    }

    /**
     * Returns a string representation of the network.
     *
     * @return the string representation of the network.
     */
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("Weights: ").append("\n");
        for (int i = 0; i < weights.size(); i++) {
            result.append("Layer ").append(i).append("\n");
            result.append(weights.get(i).toString()).append("\n");
        }
        result.append("Biases: ").append("\n");
        for (int i = 0; i < biases.size(); i++) {
            result.append("Layer ").append(i + 1).append("\n");
            result.append(biases.get(i).toString()).append("\n");
        }
        return result.toString();
    }

    /**
     * The results of testing a back propagation neural network.
     */
    public static class Result implements Network.TestResult {

        private final double accuracy;
        private final double error;
        private final List<Observation> correct;
        private final List<Observation> incorrect;

        // Compiles the data to create a result of a test on a back prop network.
        private Result(double accuracy, double error, List<Observation> correct, List<Observation> incorrect) {
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
