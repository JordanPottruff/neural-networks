package com.github.jordanpottruff.neural.models;

import com.github.jordanpottruff.jgml.MatMN;
import com.github.jordanpottruff.jgml.Vec2;
import com.github.jordanpottruff.jgml.VecN;
import com.github.jordanpottruff.neural.activations.Logistic;
import com.github.jordanpottruff.neural.common.Pair;
import com.github.jordanpottruff.neural.data.DataSet;
import com.github.jordanpottruff.neural.data.Observation;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.hamcrest.CoreMatchers.hasItem;
import static org.junit.Assert.*;

public class BackPropNetworkTest {

    private static final double EPSILON = 0.01;

    private static final InitializerStub INIT_1 = new InitializerStub(Arrays.asList(0.0, 1.0));
    private static final InitializerStub INIT_2 = new InitializerStub(Arrays.asList(0.0, 0.25));
    private static final BackPropNetwork NET_1 = new BackPropNetwork(2, new int[]{3}, new String[]{"A", "B", "C"}, new Logistic(), new Logistic(), INIT_1);
    private static final BackPropNetwork NET_2 = new BackPropNetwork(2, new int[]{3}, new String[]{"A", "B", "C"}, new Logistic(), new Logistic(), INIT_2);

    @Test
    public void testGenerateWeights() {
        MatMN expectedWeights1 = new MatMN(new double[][]{{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}});
        MatMN expectedWeights2 = new MatMN(new double[][]{{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}});

        assertTrue(expectedWeights1.equals(NET_1.weights.get(0), EPSILON));
        assertTrue(expectedWeights2.equals(NET_1.weights.get(1), EPSILON));

    }

    @Test
    public void testGenerateBiases() {
        VecN expectedBiases1 = new VecN(new double[]{0.0, 0.0, 0.0});
        VecN expectedBiases2 = new VecN(new double[]{0.0, 0.0, 0.0});

        assertTrue(expectedBiases1.equals(NET_1.biases.get(0), EPSILON));
        assertTrue(expectedBiases2.equals(NET_1.biases.get(1), EPSILON));

    }

    @Test
    public void testCalculateGradient() {
        Observation obs = new Observation(new Vec2(1.0, 2.0), "A");
        Pair<MatMN[], VecN[]> actual = NET_1.calculateGradient(obs);

        VecN expectedBiasesOutput = new VecN(new double[]{-0.071284, 0.106763, 0.148077});
        MatMN expectedWeightsOutput = new MatMN(new double[][]{
                {-0.062787, 0.094037, 0.130426},
                {-0.052113, 0.078050, 0.108253},
                {-0.062787, 0.094037, 0.130426}
        });

        VecN expectedBiasesHidden = new VecN(new double[]{0.011209, 0.015098, 0.011209});
        MatMN expectedWeightsHidden = new MatMN(new double[][]{
                {0.011209, 0.015098, 0.011209},
                {0.022419, 0.030197, 0.022419}
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
        VecN expectedHidden = new VecN(new double[]{0.880797, 0.731059, 0.880797});
        VecN expectedOutput = new VecN(new double[]{0.675038, 0.853409, 0.675038});

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

    @Test
    public void testTest() {
        Observation obs1 = new Observation(new Vec2(1, 10), "B");
        Observation obs2 = new Observation(new Vec2(1000, -1000), "C");
        DataSet data = new DataSet(Arrays.asList(obs1, obs2), new String[]{"A", "B", "C"});

        BackPropNetwork.Result result = NET_2.test(data);
        assertEquals(0.5, result.getAccuracy(), 0.01);
        assertEquals(0.37, result.getError(), 0.01);
        assertThat(result.correct(), hasItem(obs1));
        assertThat(result.incorrect(), hasItem(obs2));
    }
}
