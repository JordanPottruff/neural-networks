package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.jgml.VecN;

/**
 * Defines instances of observations. Observations are single data points that associate a series of attribute values to
 * some classification. The names/units/descriptions of these attributes are abstracted away and only position is used
 * to identify attribute values.
 */
public class Observation implements NetworkObservation {

    private final VecN attributes;
    private final String classification;

    /**
     * Creates an observation.
     * @param attributes a vector storing the attribute values for the observation.
     * @param classification the classification name for the observation.
     */
    public Observation(VecN attributes, String classification) {
        this.attributes = attributes;
        this.classification = classification;
    }

    /**
     * @inheritDoc
     */
    @Override
    public VecN getAttributes() {
        return attributes;
    }

    /**
     * @inheritDoc
     */
    @Override
    public String getClassification() {
        return classification;
    }
}
