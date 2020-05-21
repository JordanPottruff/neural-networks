package com.github.jordanpottruff.neural.models;

import com.github.jordanpottruff.jgml.MatMN;
import com.github.jordanpottruff.jgml.Vec2;
import com.github.jordanpottruff.jgml.VecN;
import com.github.jordanpottruff.neural.data.DataSet;
import com.github.jordanpottruff.neural.data.Observation;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class UtilTest {

    private static final double EPSILON = 0.001;

    @Test
    public void testComponentWiseMultiply() {
        VecN a = new VecN(new double[]{1.0, 2.0, 3.0});
        VecN b = new VecN(new double[]{-1.0, 5.0, 3.5});
        VecN aMultb = Util.componentWiseMultiply(a, b);
        VecN expected = new VecN(new double[]{-1.0, 10.0, 10.5});
        assertTrue(expected.equals(aMultb, EPSILON));
    }

    @Test
    public void testOuterProduct() {
        VecN a = new VecN(new double[]{1.0, 2.0, 3.0});
        VecN b = new VecN(new double[]{-1.0, 5.0, 3.5});
        MatMN product = Util.outerProduct(a, b);
        MatMN expected = new MatMN(new double[][]{{-1, -2, -3},
                {5, 10, 15},
                {3.5, 7, 10.5}});
        assertTrue(expected.equals(product, EPSILON));
    }

    @Test
    public void testTransposeMultiply() {
        VecN vec = new VecN(new double[]{1.0, 2.0});
        MatMN mat = new MatMN(new double[][]{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        VecN result = Util.transposeMultiply(vec, mat);
        VecN expected = new VecN(new double[]{5, 11, 17});
        assertTrue(expected.equals(result, EPSILON));
    }

    @Test
    public void testGetMiniBatches() {
        Observation obs1 = new Observation(new Vec2(1.0, 2.0), "A");
        Observation obs2 = new Observation(new Vec2(2.0, 4.0), "B");
        Observation obs3 = new Observation(new Vec2(3.0, 6.0), "C");
        Observation obs4 = new Observation(new Vec2(4.0, 8.0), "A");
        Observation obs5 = new Observation(new Vec2(5.0, 10.0), "B");
        DataSet data1 = new DataSet(Arrays.asList(obs1, obs2, obs3, obs4), new String[]{"A", "B", "C"});
        DataSet data2 = new DataSet(Arrays.asList(obs1, obs2, obs3, obs4, obs5), new String[]{"A", "B", "C"});

        List<List<Observation>> expected1 = Arrays.asList(Arrays.asList(obs1, obs2), Arrays.asList(obs3, obs4));
        List<List<Observation>> expected2 = Arrays.asList(Arrays.asList(obs1, obs2), Arrays.asList(obs3, obs4),
                Collections.singletonList(obs5));

        assertEquals(expected1, Util.getMiniBatches(data1, 2));
        assertEquals(expected2, Util.getMiniBatches(data2, 2));
    }
}
