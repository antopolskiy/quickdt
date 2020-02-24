package quickdt.predictiveModels.decisionTree;

import quickdt.predictiveModels.decisionTree.tree.ClassCounter;

public class ScoreTracker {

	private final Scorer scorer;
	private double       best         = 0;
	private double       noSplitScore = 0;
	private double       current;

	public ScoreTracker(Scorer scorer) {
		this.scorer = scorer;
	}

	/**
	 * Create a score counter and save a score value for the no split comparison.
	 * 
	 * @param scorer
	 * @param totalCounts
	 */
	public ScoreTracker(Scorer scorer, ClassCounter totalCounts) {
		this.scorer = scorer;
		this.noSplitScore = scorer.scoreSplit(totalCounts, new ClassCounter());
	}

	public double getBest() {
		return best;
	}

	public boolean isBetter(ClassCounter inCounts, ClassCounter outCounts) {
		return scorer.scoreSplit(inCounts, outCounts) > best;
	}

	public void setBest(ClassCounter inCounts, ClassCounter outCounts) {
		best = scorer.scoreSplit(inCounts, outCounts);
	}

	public boolean isBetter(InOutCounts inOutCounts) {
		return isBetter(inOutCounts.in(), inOutCounts.out());
	}

	public void setBest(InOutCounts inOutCounts) {
		setBest(inOutCounts.in(), inOutCounts.out());
	}

	public boolean noSplitIsBetter() {
		return noSplitScore >= best;
	}
}
