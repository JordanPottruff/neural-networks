package com.github.jordanpottruff.neural.data;

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
}
