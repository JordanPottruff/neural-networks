package com.github.jordanpottruff.neural.models;

import com.github.jordanpottruff.jgml.MatMN;
import com.github.jordanpottruff.jgml.Vec2;
import com.github.jordanpottruff.jgml.VecN;
import com.github.jordanpottruff.neural.common.Pair;
import com.github.jordanpottruff.neural.data.Observation;
import com.github.jordanpottruff.neural.data.RandomStub;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class BackPropNetworkTest {

    private static final double EPSILON = 0.01;

    private static final RandomStub RAND_1 = new RandomStub(Arrays.asList(0.0, 1.0));
    private static final RandomStub RAND_2 = new RandomStub(Arrays.asList(0.0, 0.25));
    private static final BackPropNetwork NET_1 = new BackPropNetwork(RAND_1, 2, new int[]{3}, new String[]{"A","B","C"}, 1.0, 1.0);
    private static final BackPropNetwork NET_2 = new BackPropNetwork(RAND_2, 2, new int[]{3}, new String[]{"A","B","C"}, 1.0, 1.0);

    @Test
    public void testGenerateRandomWeights() {
        MatMN expectedWeights1 = new MatMN(new double[][]{{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}});
        MatMN expectedWeights2 = new  MatMN(new double[][]{{0.0, 1.0, 0.0},{1.0, 0.0, 1.0},{0.0,1.0,0.0}});

        assertTrue(expectedWeights1.equals(NET_1.weights.get(0), EPSILON));
        assertTrue(expectedWeights2.equals(NET_1.weights.get(1), EPSILON));

    }

    @Test
    public void testGenerateRandomBiases() {
        VecN expectedBiases1 = new VecN(new double[]{1.0, 0.0, 1.0});
        VecN expectedBiases2 = new VecN(new double[]{0.0, 1.0, 0.0});

        assertTrue(expectedBiases1.equals(NET_1.biases.get(0), EPSILON));
        assertTrue(expectedBiases2.equals(NET_1.biases.get(1), EPSILON));

    }

    @Test
    public void testCalculateGradient() {
        Observation obs = new Observation(new Vec2(1.0, 2.0), "A");
        Pair<MatMN[], VecN[]> actual = NET_1.calculateGradient(obs);

        VecN expectedBiasesOutput = new VecN(new double[]{-0.071, 0.046, 0.148});
        MatMN expectedWeightsOutput = new MatMN(new double[][]{
                {-0.068, 0.044, 0.141},
                {-0.052, 0.034, 0.108},
                {-0.068, 0.044, 0.141}
        });

        VecN expectedBiasesHidden = new VecN(new double[]{0.002, 0.015, 0.002});
        MatMN expectedWeightsHidden = new MatMN(new double[][]{
                {0.002, 0.015, 0.002},
                {0.004, 0.03, 0.004}
        });

        assertTrue(expectedBiasesOutput.equals(actual.getValue()[1], EPSILON));
        assertTrue(expectedWeightsOutput.equals(actual.getKey()[1], EPSILON));

        assertTrue(expectedBiasesHidden.equals(actual.getValue()[0], EPSILON));
        assertTrue(expectedWeightsHidden.equals(actual.getKey()[0], EPSILON));
    }

    @Test
    public void testGetActivations() {
        VecN attributes = new VecN(new double[]{1.0, 2.0});
        List<VecN> actual = NET_1.getActivations(attributes);
        VecN expectedInput = new VecN(new double[]{1.0, 2.0});
        VecN expectedHidden = new VecN(new double[]{0.953, 0.731, 0.953});
        VecN expectedOutput = new VecN(new double[]{0.675, 0.948, 0.675});

        assertTrue(expectedInput.equals(actual.get(0), EPSILON));
        assertTrue(expectedHidden.equals(actual.get(1), EPSILON));
        assertTrue(expectedOutput.equals(actual.get(2), EPSILON));
    }

    @Test
    public void testGetExpectedOutput() {
        assertEquals(new VecN(new double[]{1.0, 0.0, 0.0}), NET_1.getExpectedOutput("A"));
        assertEquals(new VecN(new double[]{0.0, 1.0, 0.0}), NET_1.getExpectedOutput("B"));
        assertEquals(new VecN(new double[]{0.0, 0.0, 1.0}), NET_1.getExpectedOutput("C"));

    }

    @Test
    public void testClassify() {
        String classification1 = NET_1.classify(new VecN(new double[]{1, 10}));
        String classification2 = NET_2.classify(new VecN(new double[]{1000, -1000}));

        assertEquals("B", classification1);
        assertEquals("A", classification2); // Technically a tie between A and C, but the first is taken.
    }
}
