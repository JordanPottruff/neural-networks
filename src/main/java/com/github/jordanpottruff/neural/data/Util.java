package com.github.jordanpottruff.neural.data;

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
}
