package com.github.jordanpottruff.neural;

import com.github.jordanpottruff.neural.activations.Logistic;
import com.github.jordanpottruff.neural.activations.ReLU;
import com.github.jordanpottruff.neural.data.DataSet;
import com.github.jordanpottruff.neural.data.Observation;
import com.github.jordanpottruff.neural.initializers.HeInitializer;
import com.github.jordanpottruff.neural.models.BackPropNetwork;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Main {

    public static void main(String[] args) {
        String trainImageFilename = "src/main/resources/train-images.idx3-ubyte";
        String trainLabelFilename = "src/main/resources/train-labels.idx1-ubyte";
        String testImageFilename = "src/main/resources/test-images.idx3-ubyte";
        String testLabelFilename = "src/main/resources/test-labels.idx1-ubyte";
        MNISTReader trainDataReader = new MNISTReader(trainImageFilename, trainLabelFilename);
        MNISTReader testDataReader = new MNISTReader(testImageFilename, testLabelFilename);

        DataSet trainData = trainDataReader.getImageDataSet();
        DataSet testData = testDataReader.getImageDataSet();

        int inputSize = 784;
        int[] hiddenSizes = {64, 32};
        String[] classes = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        BackPropNetwork network = new BackPropNetwork(inputSize, hiddenSizes, classes, new ReLU(), new ReLU.Prime(), new Logistic(), new Logistic.Prime(), new HeInitializer());

        int n = testData.size();
        double startLearningRate = 0.03;
        double endLearningRate = 0.01;
        int epochs = 50;
        for (int i = 0; i < epochs; i++) {
            double learningRate = startLearningRate - ((startLearningRate - endLearningRate) / epochs) * i;
            System.out.println(learningRate);
            network.train(trainData, 16, learningRate);
            BackPropNetwork.Result testResult = network.test(testData);
            int numCorrect = testResult.correct().size();
            double accuracy = testResult.getAccuracy() * 100;

            System.out.println(String.format("Epoch %d: %d/%d = %.2f%%", i, numCorrect, n, accuracy));
            System.out.println(getClassCount(testResult.incorrect()));
            System.out.println(getClassCount(testResult.correct()));
            System.out.println(network.toJSON());
        }
        BackPropNetwork.Result testResult = network.test(testData);
        int numCorrect = testResult.correct().size();
        double accuracy = testResult.getAccuracy() * 100;
        System.out.println(String.format("Final:  %d/%d = %.2f%%", numCorrect, n, accuracy));

        System.out.println(network);
        System.out.println(network.toJSON());
    }

    public static Map<String, Integer> getClassCount(List<Observation> incorrect) {
        Map<String, Integer> counts = new HashMap<>();
        for (int i = 0; i < 10; i++) {
            counts.put(Integer.toString(i), 0);
        }

        for (Observation obs : incorrect) {
            counts.put(obs.getClassification(), counts.get(obs.getClassification()) + 1);
        }
        return counts;
    }
}
