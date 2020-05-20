package com.github.jordanpottruff.neural;

import com.github.jordanpottruff.jgml.Vec2;
import com.github.jordanpottruff.neural.common.Pair;
import com.github.jordanpottruff.neural.data.DataSet;
import com.github.jordanpottruff.neural.data.Observation;
import com.github.jordanpottruff.neural.models.BackPropNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

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
        BackPropNetwork network = new BackPropNetwork(inputSize, hiddenSizes, classes);

        int n = testData.size();
        for(int i=0; i<50; i++) {
            network.train(trainData, 16, 1);
            BackPropNetwork.Result testResult = network.test(testData);
            int numCorrect = testResult.correct().size();
            double accuracy = testResult.getAccuracy()*100;

            System.out.println(String.format("Epoch %d: %d/%d = %.2f%%", i, numCorrect, n, accuracy));
        }
        BackPropNetwork.Result testResult = network.test(testData);
        int numCorrect = testResult.correct().size();
        double accuracy = testResult.getAccuracy()*100;
        System.out.println(String.format("Final:  %d/%d = %.2f%%", numCorrect, n, accuracy));
    }
}
