package com.github.jordanpottruff.neural.models;

import com.github.jordanpottruff.jgml.MatMN;
import com.github.jordanpottruff.jgml.Vec2;
import com.github.jordanpottruff.jgml.VecN;
import com.github.jordanpottruff.neural.common.Pair;
import com.github.jordanpottruff.neural.data.Observation;
import com.github.jordanpottruff.neural.data.RandomStub;
import com.github.jordanpottruff.neural.models.BackPropNetwork;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class BackPropNetworkTest {

    private static final double EPSILON = 0.01;

    private static final RandomStub RAND = new RandomStub(Arrays.asList(0.0, 1.0));
    private static final BackPropNetwork NET = new BackPropNetwork(RAND, 2, new int[]{3}, new String[]{"A","B","C"}, 1.0, 1.0);

    @Test
    public void testGenerateRandomWeights() {
        MatMN expectedWeights1 = new MatMN(new double[][]{{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}});
        MatMN expectedWeights2 = new  MatMN(new double[][]{{0.0, 1.0, 0.0},{1.0, 0.0, 1.0},{0.0,1.0,0.0}});

        assertTrue(expectedWeights1.equals(NET.weights.get(0), EPSILON));
        assertTrue(expectedWeights2.equals(NET.weights.get(1), EPSILON));

    }

    @Test
    public void testGenerateRandomBiases() {
        VecN expectedBiases1 = new VecN(new double[]{1.0, 0.0, 1.0});
        VecN expectedBiases2 = new VecN(new double[]{0.0, 1.0, 0.0});

        assertTrue(expectedBiases1.equals(NET.biases.get(0), EPSILON));
        assertTrue(expectedBiases2.equals(NET.biases.get(1), EPSILON));

    }

    @Test
    public void testCalculateGradient() {
        Observation obs = new Observation(new Vec2(1.0, 2.0), "A");
        Pair<MatMN[], VecN[]> actual = NET.calculateGradient(obs);

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
        List<VecN> actual = NET.getActivations(attributes);
        VecN expectedInput = new VecN(new double[]{1.0, 2.0});
        VecN expectedHidden = new VecN(new double[]{0.953, 0.731, 0.953});
        VecN expectedOutput = new VecN(new double[]{0.675, 0.948, 0.675});

        assertTrue(expectedInput.equals(actual.get(0), EPSILON));
        assertTrue(expectedHidden.equals(actual.get(1), EPSILON));
        assertTrue(expectedOutput.equals(actual.get(2), EPSILON));
    }

    @Test
    public void testGetExpectedOutput() {
        assertEquals(new VecN(new double[]{1.0, 0.0, 0.0}), NET.getExpectedOutput("A"));
        assertEquals(new VecN(new double[]{0.0, 1.0, 0.0}), NET.getExpectedOutput("B"));
        assertEquals(new VecN(new double[]{0.0, 0.0, 1.0}), NET.getExpectedOutput("C"));

    }
}
