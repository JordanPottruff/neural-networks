package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.jgml.MatMN;
import com.github.jordanpottruff.jgml.VecN;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class UtilTest {

    private static final double EPSILON = 0.001;

    @Test
    public void testApplyLogistic() {
        VecN a = new VecN(new double[]{1.0, 2.0, 3.0});
        VecN a_logistic = Util.applyLogistic(a);
        VecN expected = new VecN(new double[]{0.731, 0.881, 0.953});
        assertTrue(expected.equals(a_logistic, EPSILON));
    }

    @Test
    public void testApplyLogisticPrime() {
        VecN a = new VecN(new double[]{1.0, 2.0, 3.0});
        VecN a_logistic_prime = Util.applyLogisticPrime(a);
        VecN expected = new VecN(new double[]{0.0, -2.0, -6.0});
        assertTrue(expected.equals(a_logistic_prime, EPSILON));
    }

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
        MatMN expected = new MatMN(new double[][]{  {-1, -2, -3},
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
}
