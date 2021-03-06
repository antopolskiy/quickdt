package quickdt.predictiveModels.decisionTree.scorers;

import java.io.Serializable;
import java.util.Map;

import quickdt.predictiveModels.decisionTree.Scorer;
import quickdt.predictiveModels.decisionTree.tree.ClassCounter;

/**
 * Created by chrisreeves on 6/24/14.
 */
public class GiniImpurityScorer implements Scorer {
	@Override
	public double scoreSplit(ClassCounter a, ClassCounter b) {
		ClassCounter parent = ClassCounter.merge(a, b);
		double parentGiniIndex = getGiniIndex(parent);
		double aGiniIndex = getGiniIndex(a) * a.getTotal() / parent.getTotal();
		double bGiniIndex = getGiniIndex(b) * b.getTotal() / parent.getTotal();
		return parentGiniIndex - aGiniIndex - bGiniIndex;
	}

	private double getGiniIndex(ClassCounter cc) {
		double sum = 0.0d;
		for (Map.Entry<Serializable, Double> e : cc.getCounts().entrySet()) {
			double error = (cc.getTotal() > 0) ? e.getValue() / cc.getTotal() : 0;
			sum += error * error;
		}
		return 1.0d - sum;
	}

	@Override
	public String toString() {
		return "GiniImpurity";
	}
}
