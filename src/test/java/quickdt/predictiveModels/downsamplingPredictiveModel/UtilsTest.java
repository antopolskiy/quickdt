package quickdt.predictiveModels.downsamplingPredictiveModel;

import org.junit.Assert;
import org.junit.Test;

public class UtilsTest {

  @Test
  public void correctProbabilityInputZeroZeroOutputZero() {
    Assert.assertEquals(0.0, Utils.correctProbability(0.0, 0.0), 0.0);
  }
}
