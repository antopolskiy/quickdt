package quickdt.predictiveModels.decisionTree.scorers;

import org.testng.Assert;
import org.testng.annotations.Test;

import quickdt.predictiveModels.decisionTree.tree.ClassCounter;

/**
 * Created by ian on 2/27/14.
 */
public class MSEScorerTest {
	@Test
	public void simpleTest() {
		ClassCounter a = new ClassCounter();
		a.addClassification("a", 4);
		a.addClassification("b", 9);
		a.addClassification("c", 1);
		ClassCounter b = new ClassCounter();
		b.addClassification("a", 5);
		b.addClassification("b", 9);
		b.addClassification("c", 6);
		MSEScorer mseScorer = new MSEScorer(MSEScorer.CrossValidationCorrection.FALSE);
		Assert.assertTrue(Math.abs(mseScorer.scoreSplit(a, b) - 0.021776929) < 0.000000001);
	}
}
