package quickdt.predictiveModels.calibratedPredictiveModel;

import org.junit.Assert;
import org.junit.Test;

import quickdt.predictiveModels.calibratedPredictiveModel.PAVCalibrator.Observation;

import java.util.ArrayList;

public class PAVCalibratorTest {

  @Test
  public void constructorInputPositiveZeroZeroOutputNotNull() throws Exception {
    final Observation observation = new Observation(1.0, 0.0, 0.0);
    Assert.assertNotNull(observation);
    Assert.assertEquals(0.0, observation.weight, 0.0);
    Assert.assertEquals(0.0, observation.output, 0.0);
    Assert.assertEquals(1.0, observation.input, 0.0);
  }

  @Test
  public void constructorInputZeroZeroOutputNotNull() throws Exception {
    final Observation observation = new Observation(0.0, 0.0);
    Assert.assertNotNull(observation);
    Assert.assertEquals(1.0, observation.weight, 0.0);
    Assert.assertEquals(0.0, observation.output, 0.0);
    Assert.assertEquals(0.0, observation.input, 0.0);
  }

  @Test
  public void newWeightlessInputZeroZeroOutputNotNull() throws Exception {
    final Observation observation = Observation.newWeightless(0.0, 0.0);
    Assert.assertNotNull(observation);
    Assert.assertEquals(0.0, observation.weight, 0.0);
    Assert.assertEquals(0.0, observation.output, 0.0);
    Assert.assertEquals(0.0, observation.input, 0.0);
  }

  @Test(expected = IllegalArgumentException.class)
  public void constructorInputNotNullNegativeOutputIllegalArgumentException() {
    final ArrayList<Observation> predictions = new ArrayList<Observation>();
    new PAVCalibrator(predictions, -2_147_221_504);
  }
}
