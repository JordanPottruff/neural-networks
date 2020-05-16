package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.jgml.MatMN;
import com.github.jordanpottruff.jgml.VecN;

class Util {

    static VecN applyLogistic(VecN vec) {
        double[] values = vec.toArray();
        for(int i=0; i<values.length; i++) {
            values[i] = 1.0 / (1.0 + Math.exp(-values[i]));
        }
        return new VecN(values);
    }

    static VecN applyLogisticPrime(VecN vec) {
        double[] values = vec.toArray();
        for(int i=0; i<values.length; i++) {
            values[i] = values[i] * (1-values[i]);
        }
        return new VecN(values);
    }

    static VecN componentWiseMultiply(VecN a, VecN b) {
        double[] a_values = a.toArray();
        double[] b_values = b.toArray();
        for(int i=0; i<a_values.length; i++) {
            a_values[i] *= b_values[i];
        }
        return new VecN(a_values);
    }

    static MatMN outerProduct(VecN a, VecN b) {
        double[] a_values = a.toArray();
        double[] b_values = b.toArray();
        double[][] product = new double[b_values.length][a_values.length];

        for(int col=0; col < b_values.length; col++) {
            for(int row=0; row < a_values.length; row++) {
                product[col][row] = b_values[col] * a_values[row];
            }
        }
        return new MatMN(product);
    }
}
