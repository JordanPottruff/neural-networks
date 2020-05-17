package com.github.jordanpottruff.neural.data;

import com.github.jordanpottruff.jgml.Vec2;
import com.github.jordanpottruff.neural.common.Pair;
import org.junit.Test;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

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

    @Test
    public void testSplit() {
        Pair<DataSet, DataSet> actual1 = DATA_SET_1.split(0.5);
        List<Observation> expectedLeft1 = Arrays.asList(OBS_1, OBS_2);
        List<Observation> expectedRight1 = Arrays.asList(OBS_3, OBS_4);

        Pair<DataSet, DataSet> actual2 = DATA_SET_1.split(0.75);
        List<Observation> expectedLeft2 = Arrays.asList(OBS_1, OBS_2, OBS_3);
        List<Observation> expectedRight2 = Collections.singletonList(OBS_4);

        assertEquals(expectedLeft1, actual1.getKey().getAllObservations());
        assertEquals(expectedRight1, actual1.getValue().getAllObservations());

        assertEquals(expectedLeft2, actual2.getKey().getAllObservations());
        assertEquals(expectedRight2, actual2.getValue().getAllObservations());
    }
}
