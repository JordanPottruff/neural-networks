package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.jgml.MatMN;
import com.github.jordanpottruff.jgml.VecN;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertTrue;

public class BackPropNetworkTest {

    static private final double EPSILON = 0.01;

    @Test
    public void testGenerateRandomWeights() {
        List<Double> weightValues = Arrays.asList(0.0, 1.0);
        RandomStub random = new RandomStub(weightValues);
        BackPropNetwork net = new BackPropNetwork(random, 2, new int[]{3}, new String[]{"A","B","C"}, 1.0, 1.0);

        MatMN expectedWeights1 = new MatMN(new double[][]{{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}});
        MatMN expectedWeights2 = new  MatMN(new double[][]{{0.0, 1.0, 0.0},{1.0, 0.0, 1.0},{0.0,1.0,0.0}});

        assertTrue(expectedWeights1.equals(net.weights.get(0), EPSILON));
        assertTrue(expectedWeights2.equals(net.weights.get(1), EPSILON));

    }

    @Test
    public void testGenerateRandomBiases() {
        List<Double> biasValues = Arrays.asList(0.0, 1.0);
        RandomStub random = new RandomStub(biasValues);
        BackPropNetwork net = new BackPropNetwork(random, 2, new int[]{3}, new String[]{"A","B","C"}, 1.0, 1.0);

        VecN expectedBiases1 = new VecN(new double[]{1.0, 0.0, 1.0});
        VecN expectedBiases2 = new VecN(new double[]{0.0, 1.0, 0.0});

        assertTrue(expectedBiases1.equals(net.biases.get(0), EPSILON));
        assertTrue(expectedBiases2.equals(net.biases.get(1), EPSILON));

    }
}
