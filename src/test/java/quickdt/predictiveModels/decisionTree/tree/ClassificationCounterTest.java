package quickdt.predictiveModels.decisionTree.tree;

import org.junit.Assert;
import org.junit.Test;

public class ClassificationCounterTest {

  @Test
  public void equalsInputNotNullOutputFalse() {
    final ClassificationCounter classificationCounter = new ClassificationCounter();
    final ClassificationCounter o = new ClassificationCounter();
    Assert.assertFalse(classificationCounter.equals(o));
  }

  @Test
  public void equalsInputNullOutputFalse() {
    final ClassificationCounter classificationCounter = new ClassificationCounter();
    Assert.assertFalse(classificationCounter.equals(null));
  }

  @Test
  public void getCountInputFalseOutputZero() {
    final ClassificationCounter classificationCounter = new ClassificationCounter();
    Assert.assertEquals(0.0, classificationCounter.getCount(false), 0.0);
  }

  @Test
  public void getTotalOutputZero() {
    final ClassificationCounter classificationCounter = new ClassificationCounter();
    Assert.assertEquals(0.0, classificationCounter.getTotal(), 0.0);
  }
}
