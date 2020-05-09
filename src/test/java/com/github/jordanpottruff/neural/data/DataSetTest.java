package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.jgml.Vec2;
import org.junit.Test;
import java.util.Arrays;

import static org.junit.Assert.*;


public class DataSetTest {

    private final String[] CLASSES = {"ClassA", "ClassB", "ClassC"};

    private final Observation OBS_1 = new Observation(new Vec2(1.0, 2.0), "ClassA");
    private final Observation OBS_2 = new Observation(new Vec2(0.0, -1.0), "ClassB");
    private final Observation OBS_3 = new Observation(new Vec2(1.0, -3.0), "ClassB");
    private final Observation OBS_4 = new Observation(new Vec2(5.0, 10.0), "classC");

    private final DataSet DATA_SET_1 = new DataSet(Arrays.asList(OBS_1, OBS_2, OBS_3, OBS_4), CLASSES);

    @Test
    public void testSize() {
        assertEquals(DATA_SET_1.size(), 4);
    }

    @Test
    public void testGetClasses() {
        assertArrayEquals(DATA_SET_1.getClasses(), CLASSES);
    }

    @Test
    public void testGetObservation() {
        assertEquals(DATA_SET_1.getObservation(0), OBS_1);
        assertEquals(DATA_SET_1.getObservation(1), OBS_2);
        assertEquals(DATA_SET_1.getObservation(2), OBS_3);
        assertEquals(DATA_SET_1.getObservation(3), OBS_4);
    }

    @Test
    public void testGetAllObservations() {
        assertEquals(DATA_SET_1.getAllObservations(), Arrays.asList(OBS_1, OBS_2, OBS_3, OBS_4));
    }
}
