
import com.github.jordanpottruff.jgml.Vec3;
import com.github.jordanpottruff.jgml.VecN;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ObservationTest {

    @Test
    public void test_getAttributes() {
        Observation obs = new Observation(new Vec3(1.0, 2.0, 3.0), "classA");
        VecN actual = obs.getAttributes();

        assertTrue(actual.equals(new Vec3(1.0, 2.0, 3.0), 0.001));
    }

    @Test
    public void test_getClassification() {
        Observation obs = new Observation(new Vec3(1.0, 2.0, 3.0), "classA");
        String actual = obs.getClassification();

        assertEquals(actual, "classA");
    }
}
