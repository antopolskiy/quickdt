package quickdt;

import org.testng.Assert;
import org.testng.annotations.Test;

import quickdt.predictiveModels.decisionTree.tree.ClassCounter;

/**
 * Created by ian on 2/27/14.
 */
public class ClassCounterTest {

	@Test
	public void testAdd() {
		ClassCounter a = new ClassCounter();
		a.addClassification("dog", 1.0);
		a.addClassification("cat", 0.5);
		ClassCounter b = new ClassCounter();
		b.addClassification("dog", 0.5);
		b.addClassification("cat", 1.0);
		ClassCounter c = a.add(b);
		Assert.assertEquals(c.getCount("dog"), 1.5);
		Assert.assertEquals(c.getCount("cat"), 1.5);
	}

	@Test
	public void testSubtract() {
		ClassCounter a = new ClassCounter();
		a.addClassification("dog", 1.0);
		a.addClassification("cat", 2.5);
		ClassCounter b = new ClassCounter();
		b.addClassification("dog", 0.5);
		b.addClassification("cat", 1.0);
		ClassCounter c = a.subtract(b);
		Assert.assertEquals(c.getCount("dog"), 0.5);
		Assert.assertEquals(c.getCount("cat"), 1.5);
	}

	@Test
	public void testMerge() {
		ClassCounter a = new ClassCounter();
		a.addClassification("dog", 1.0);
		a.addClassification("cat", 0.5);
		ClassCounter b = new ClassCounter();
		b.addClassification("dog", 0.5);
		b.addClassification("cat", 1.0);
		ClassCounter merged = ClassCounter.merge(a, b);
		Assert.assertEquals(merged.getTotal(), 3.0);
		Assert.assertEquals(merged.getCount("dog"), 1.5);
		Assert.assertEquals(merged.getCount("cat"), 1.5);
	}
}
