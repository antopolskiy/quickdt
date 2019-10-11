package quickdt.crossValidation.movingAverages;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;

public class ArithmeticAverageTest {

  @Test
  public void constructorOutputNotNull() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    Assert.assertNotNull(arithmeticAverage);
    Assert.assertEquals(0.0, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInputNotNullOutputNaN() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    final ArrayList<Double> values = new ArrayList<Double>();
    Assert.assertEquals(Double.NaN, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(Double.NaN, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInputNotNullOutputNegative() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    final ArrayList<Double> values = new ArrayList<Double>();
    values.add(0x1.b3fffff600004p+1021 /* 3.82712e+307 */);
    values.add(-0x1.19fffffb00003p+1022 /* -4.95068e+307 */);
    values.add(0x1.0000000000004p+1020 /* 1.12356e+307 */);
    values.add(0x1.000100008p+695 /* 1.64382e+209 */);
    values.add(-0x1.0000080008p+716 /* -3.44728e+215 */);
    values.add(-0x1.c17ffffff7fdfp+763 /* -8.51876e+229 */);
    Assert.assertEquals(-0x1.2baaaaaaa5555p+761 /* -1.41979e+229 */, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(-0x1.2baaaaaaa5555p+761 /* -1.41979e+229 */, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInputNotNullOutputPositive() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    final ArrayList<Double> values = new ArrayList<Double>();
    values.add(0x1.aaaaad8p+30 /* 1.78957e+09 */);
    Assert.assertEquals(0x1.aaaaad8p+30 /* 1.78957e+09 */, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(0x1.aaaaad8p+30 /* 1.78957e+09 */, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInputNotNullOutputPositive1() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    final ArrayList<Double> values = new ArrayList<Double>();
    values.add(4.0);
    Assert.assertEquals(4.0, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(4.0, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInput0OutputNaN() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    arithmeticAverage.average = 0.0;
    final ArrayList<Double> values = new ArrayList<Double>();
    Assert.assertEquals(Double.NaN, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(Double.NaN, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInput0OutputNegativeInfinity() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    arithmeticAverage.average = -0x1.ce1p+69 /* -1.06544e+21 */;
    final ArrayList<Double> values = new ArrayList<Double>();
    Assert.assertEquals(Double.NEGATIVE_INFINITY, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(Double.NEGATIVE_INFINITY, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInput0OutputNegativeInfinity1() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    arithmeticAverage.average = -0x1.2p-629 /* -5.04993e-190 */;
    final ArrayList<Double> values = new ArrayList<Double>();
    Assert.assertEquals(Double.NEGATIVE_INFINITY, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(Double.NEGATIVE_INFINITY, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInput0OutputPositiveInfinity() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    arithmeticAverage.average = 0x1.ffffffecp+30 /* 2.14748e+09 */;
    final ArrayList<Double> values = new ArrayList<Double>();
    Assert.assertEquals(Double.POSITIVE_INFINITY, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(Double.POSITIVE_INFINITY, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInput0OutputPositiveInfinity1() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    arithmeticAverage.average = 0x1.0000000001p+1014 /* 1.75556e+305 */;
    final ArrayList<Double> values = new ArrayList<Double>();
    Assert.assertEquals(Double.POSITIVE_INFINITY, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(Double.POSITIVE_INFINITY, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInput1OutputNegative() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    arithmeticAverage.average = -0x1.0000000080001p+930 /* -9.07603e+279 */;
    final ArrayList<Double> values = new ArrayList<Double>();
    values.add(0x1.0000200000002p+897 /* 1.05659e+270 */);
    Assert.assertEquals(-0x1p+930 /* -9.07603e+279 */, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(-0x1p+930 /* -9.07603e+279 */, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInput1OutputNegative1() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    arithmeticAverage.average = 0x1.ffffffecp+30 /* 2.14748e+09 */;
    final ArrayList<Double> values = new ArrayList<Double>();
    values.add(-0x1p+31 /* -2.14748e+09 */);
    Assert.assertEquals(-5.0, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(-5.0, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInput1OutputPositive() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    arithmeticAverage.average = -0x1.2p-629 /* -5.04993e-190 */;
    final ArrayList<Double> values = new ArrayList<Double>();
    values.add(0x1p-511 /* 1.49167e-154 */);
    Assert.assertEquals(0x1p-511 /* 1.49167e-154 */, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(0x1p-511 /* 1.49167e-154 */, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInput1OutputPositive1() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    arithmeticAverage.average = -0x1.ffffffffffe8p-1009 /* -3.64556e-304 */;
    final ArrayList<Double> values = new ArrayList<Double>();
    values.add(0x1.000000000008p-1008 /* 3.64556e-304 */);
    Assert.assertEquals(0x0.00000005p-1022 /* 2.59033e-317 */, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(0x0.00000005p-1022 /* 2.59033e-317 */, arithmeticAverage.average, 0.0);
  }

  @Test
  public void getAverageInput4OutputPositive() {
    final ArithmeticAverage arithmeticAverage = new ArithmeticAverage();
    arithmeticAverage.average = 0x1.4410033fe0012p-955 /* 4.15665e-288 */;
    final ArrayList<Double> values = new ArrayList<Double>();
    values.add(-0x1.0000cff7ffe7p-961 /* -5.13073e-290 */);
    values.add(-0x1.00000000180a4p-967 /* -8.01667e-292 */);
    values.add(0x1.c0000007fffep-955 /* 5.74635e-288 */);
    values.add(-0x1.781ffde4p-954 /* -9.64887e-288 */);
    Assert.assertEquals(0x1.f80087ffffcp-962 /* 5.05053e-290 */, arithmeticAverage.getAverage(values), 0.0);
    Assert.assertEquals(0x1.f80087ffffcp-962 /* 5.05053e-290 */, arithmeticAverage.average, 0.0);
  }
}
