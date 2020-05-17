package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.jgml.MatMN;
import com.github.jordanpottruff.jgml.Vec2;
import com.github.jordanpottruff.jgml.VecN;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

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
    public void testGetActivations() {
        VecN attributes = new Vec2(1.0, 2.0);
        List<VecN> actual = NET.getActivations(attributes);
        VecN expectedHidden = new VecN(new double[]{0.953, 0.731, 0.953});
        VecN expectedOutput = new VecN(new double[]{0.675, 0.948, 0.675});

        assertTrue(expectedHidden.equals(actual.get(0), EPSILON));
        assertTrue(expectedOutput.equals(actual.get(1), EPSILON));
    }
}
