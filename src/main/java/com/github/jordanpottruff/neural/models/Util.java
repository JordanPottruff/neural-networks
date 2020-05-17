package com.github.jordanpottruff.neural.models;

import com.github.jordanpottruff.jgml.MatMN;
import com.github.jordanpottruff.jgml.VecN;
import com.github.jordanpottruff.neural.data.DataSet;
import com.github.jordanpottruff.neural.data.Observation;

import java.util.ArrayList;
import java.util.List;

class Util {

    static VecN applyLogistic(VecN vec) {
        double[] values = vec.toArray();
        for (int i = 0; i < values.length; i++) {
            values[i] = 1.0 / (1.0 + Math.exp(-values[i]));
        }
        return new VecN(values);
    }

    static VecN applyLogisticPrime(VecN vec) {
        double[] values = vec.toArray();
        for (int i = 0; i < values.length; i++) {
            values[i] = values[i] * (1 - values[i]);
        }
        return new VecN(values);
    }

    static VecN componentWiseMultiply(VecN a, VecN b) {
        double[] a_values = a.toArray();
        double[] b_values = b.toArray();
        for (int i = 0; i < a_values.length; i++) {
            a_values[i] *= b_values[i];
        }
        return new VecN(a_values);
    }

    static MatMN outerProduct(VecN a, VecN b) {
        double[] a_values = a.toArray();
        double[] b_values = b.toArray();
        double[][] product = new double[b_values.length][a_values.length];

        for (int col = 0; col < b_values.length; col++) {
            for (int row = 0; row < a_values.length; row++) {
                product[col][row] = b_values[col] * a_values[row];
            }
        }
        return new MatMN(product);
    }

    static VecN transposeMultiply(VecN vec, MatMN mat) {
        double[] result = new double[mat.cols()];
        for (int col = 0; col < mat.cols(); col++) {
            for (int row = 0; row < vec.size(); row++) {
                result[col] += vec.get(row) * mat.get(row, col);
            }
        }
        return new VecN(result);
    }

    static List<List<Observation>> getMiniBatches(DataSet dataSet, int miniBatchSize) {
        List<List<Observation>> batches = new ArrayList<>();
        List<Observation> currentBatch = new ArrayList<>();
        for(int i=0; i<dataSet.size(); i++) {
            currentBatch.add(dataSet.getObservation(i));
            if((i+1)%miniBatchSize == 0) {
                batches.add(currentBatch);
                currentBatch = new ArrayList<>();
            }
        }
        if(currentBatch.size() != 0) {
            batches.add(currentBatch);
        }
        return batches;
    }
}
