import com.github.jordanpottruff.jgml.VecN;

import java.util.List;

/**
 * Defines a data set to be used in a neural network. A data set is composed of a series of observations.
 */
public interface NetworkDataSet {

    /**
     * Returns the number of observations in the data set.
     *
     * @return the number of observations in the data set.
     */
    public int size();

    public String[] getClasses();

    /**
     * Returns the ith observation in the data set.
     *
     * @param i the index of the observation to be returned.
     * @return the observation at index i.
     */
    public NetworkObservation getObservation(int i);

    /**
     * Returns a list of all the observations in the data set.
     *
     * @return the list of observations.
     */
    public List<? extends NetworkObservation> getAllObservations();
}
