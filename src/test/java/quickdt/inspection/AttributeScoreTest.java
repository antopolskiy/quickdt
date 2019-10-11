package quickdt.inspection;

import org.junit.Assert;
import org.junit.Test;

public class AttributeScoreTest {

  @Test
  public void compareToInputNotNullOutputZero() {
    final AttributeScore attributeScore = new AttributeScore("foo", 0.0);
    final AttributeScore o = new AttributeScore("foo", 0.0);
    Assert.assertEquals(0, attributeScore.compareTo(o));
  }

  @Test
  public void getAttributeOutputNotNull() {
    final AttributeScore attributeScore = new AttributeScore("foo", 0.0);
    Assert.assertEquals("foo", attributeScore.getAttribute());
  }

  @Test
  public void getScoreOutputZero() {
    final AttributeScore attributeScore = new AttributeScore("foo", 0.0);
    Assert.assertEquals(0.0, attributeScore.getScore(), 0.0);
  }

  @Test
  public void toStringOutputNotNull() {
    final AttributeScore attributeScore = new AttributeScore("foo", 1.0);
    Assert.assertEquals("AttributeScore{attribute='foo', score=1.0}", attributeScore.toString());
  }
}
