package quickdt.predictiveModels.downsamplingPredictiveModel;

import junit.framework.Assert;
import org.testng.annotations.Test;
import quickdt.data.Attributes;
import quickdt.data.HashMapAttributes;
import quickdt.predictiveModels.PredictiveModel;

import static org.mockito.Mockito.*;

/**
 * Created by ian on 4/24/14.
 */
public class DownsamplingPredictiveModelTest {
    @Test
    public void simpleTest() {
        final PredictiveModel wrappedPredictiveModel = mock(PredictiveModel.class);
        when(wrappedPredictiveModel.getProbability(any(Attributes.class), eq(Boolean.TRUE))).thenReturn(0.5);
        DownsamplingPredictiveModel downsamplingPredictiveModel = new DownsamplingPredictiveModel(wrappedPredictiveModel, Boolean.FALSE, Boolean.TRUE, 0.9);
        double corrected = downsamplingPredictiveModel.getProbability(new HashMapAttributes(), Boolean.TRUE);
        double error = Math.abs(corrected - 0.1/1.1);
        Assert.assertTrue(String.format("Error (%s) should be negligible", error), error < 0.0000001);
    }

    @Test
    public void getDropProbabilityOutputZero() {
      final DownsamplingPredictiveModel downsamplingPredictiveModel2 = new DownsamplingPredictiveModel(null, false, false, 0.0);
      final DownsamplingPredictiveModel downsamplingPredictiveModel1 = new DownsamplingPredictiveModel(downsamplingPredictiveModel2, false, false, 0.0);
      final DownsamplingPredictiveModel downsamplingPredictiveModel = new DownsamplingPredictiveModel(downsamplingPredictiveModel1, false, false, 0.0);
      Assert.assertEquals(0.0, downsamplingPredictiveModel.getDropProbability(), 0.0);
    }

    @Test
    public void getMajorityClassificationOutputFalse() {
      final DownsamplingPredictiveModel downsamplingPredictiveModel2 = new DownsamplingPredictiveModel(null, false, false, 0.0);
      final DownsamplingPredictiveModel downsamplingPredictiveModel1 = new DownsamplingPredictiveModel(downsamplingPredictiveModel2, false, false, 0.0);
      final DownsamplingPredictiveModel downsamplingPredictiveModel = new DownsamplingPredictiveModel(downsamplingPredictiveModel1, false, false, 0.0);
      Assert.assertFalse((boolean) downsamplingPredictiveModel.getMajorityClassification());
    }
}
