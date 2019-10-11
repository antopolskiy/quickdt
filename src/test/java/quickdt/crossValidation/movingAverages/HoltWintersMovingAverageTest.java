package quickdt.crossValidation.movingAverages;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;

public class HoltWintersMovingAverageTest {

  @Test
  public void constructorInputZeroZeroOutputNotNull() {
    final HoltWintersMovingAverage holtWintersMovingAverage = new HoltWintersMovingAverage(0.0, 0.0);
    Assert.assertNotNull(holtWintersMovingAverage);
    Assert.assertEquals(0.0, holtWintersMovingAverage.average, 0.0);

  }

  @Test(expected = IndexOutOfBoundsException.class)
  public void getAverageInputNotNullOutputIndexOutOfBoundsException() {
    final HoltWintersMovingAverage holtWintersMovingAverage = new HoltWintersMovingAverage(0.0, 0.0);
    final ArrayList<Double> values = new ArrayList<Double>();
    holtWintersMovingAverage.getAverage(values);
  }

  @Test
  public void getAverageInputNotNullOutputNaN() {
    final HoltWintersMovingAverage holtWintersMovingAverage = new HoltWintersMovingAverage(0.0, 0x1p+61 /* 2.30584e+18 */);
    holtWintersMovingAverage.setAlpha(-0x1.8da62efe2bf4ep+810 /* -1.06061e+244 */);
    holtWintersMovingAverage.setBeta(-0x1.ffffffefdff71p+28 /* -5.36871e+08 */);
    final ArrayList<Double> values = new ArrayList<Double>();
    values.add(0x1.f4000000004p+960 /* 1.90338e+289 */);
    values.add(0x1.2000007fffc13p+965 /* 3.50831e+290 */);
    values.add(0x1.4bee3b6c54cp-986 /* 1.98258e-297 */);
    values.add(0x0.f914ffad61564p-1022 /* 2.16494e-308 */);
    values.add(0x1.499e00ab78269p-332 /* 1.47167e-100 */);
    values.add(-0.0);
    values.add(0.0);
    values.add(-0x1.0126bc15c5a45p-810 /* -1.47113e-244 */);
    values.add(0x1.0e36d67p-1022 /* 2.34862e-308 */);
    values.add(-0.0);
    Assert.assertEquals(Double.NaN, holtWintersMovingAverage.getAverage(values), 0.0);
  }

  @Test
  public void getAverageInputNotNullOutputNegative() {
    final HoltWintersMovingAverage holtWintersMovingAverage = new HoltWintersMovingAverage(-1.0, 9.0);
    holtWintersMovingAverage.setAlpha(-1.0);
    holtWintersMovingAverage.setBeta(1024.0);
    final ArrayList<Double> values = new ArrayList<Double>();
    values.add(-25215.0);
    values.add(0x1.7ffe77p+30 /* 1.61059e+09 */);
    values.add(0x1.6dbp+24 /* 2.39657e+07 */);
    Assert.assertEquals(-0x1.6e74fep+24 /* -2.40161e+07 */, holtWintersMovingAverage.getAverage(values), 0.0);
  }

  @Test
  public void getAverageInputNotNullOutputPositive() {
    final HoltWintersMovingAverage holtWintersMovingAverage = new HoltWintersMovingAverage(-1.0, 9.0);
    holtWintersMovingAverage.setBeta(1024.0);
    final ArrayList<Double> values = new ArrayList<Double>();
    values.add(-256.0);
    values.add(0x1.7ffffcp+30 /* 1.61061e+09 */);
    Assert.assertEquals(0x1.7ffffcp+30 /* 1.61061e+09 */, holtWintersMovingAverage.getAverage(values), 0.0);
  }
}
