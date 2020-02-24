package quickdt.predictiveModels.decisionTree.scorers;

import org.testng.Assert;
import org.testng.annotations.Test;

import quickdt.predictiveModels.decisionTree.tree.ClassCounter;

public class GiniImpurityScorerTest {
	@Test
	public void sameClassificationTest() {
		ClassCounter a = new ClassCounter();
		a.addClassification("a", 4);
		ClassCounter b = new ClassCounter();
		b.addClassification("a", 4);
		GiniImpurityScorer scorer = new GiniImpurityScorer();
		Assert.assertEquals(scorer.scoreSplit(a, b), 0.0);
	}

	@Test
	public void diffClassificationTest() {
		ClassCounter a = new ClassCounter();
		a.addClassification("a", 4);
		ClassCounter b = new ClassCounter();
		b.addClassification("b", 4);
		GiniImpurityScorer scorer = new GiniImpurityScorer();
		Assert.assertEquals(scorer.scoreSplit(a, b), 0.5);
	}
}
