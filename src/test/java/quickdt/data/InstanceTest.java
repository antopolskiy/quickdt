package quickdt.data;

import org.junit.Assert;
import org.junit.Test;

import java.io.Serializable;

public class InstanceTest {

  @Test
  public void createInputFalse0OutputNotNull() {
    final Serializable[] inputs = { };
    final Instance instance = Instance.create(false, inputs);
    Assert.assertNotNull(instance);
    Assert.assertEquals(0, instance.index);
  }

  @Test
  public void createInputFalseZero0OutputNotNull() {
    final Serializable[] inputs = { };
    final Instance instance = Instance.create(false, 0.0, inputs);
    Assert.assertNotNull(instance);
    Assert.assertEquals(0, instance.index);
  }
}
