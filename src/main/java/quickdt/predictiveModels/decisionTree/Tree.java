package quickdt.predictiveModels.decisionTree;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.collect.Maps;

import quickdt.data.Attributes;
import quickdt.predictiveModels.PredictiveModel;
import quickdt.predictiveModels.decisionTree.tree.ClassificationCounter;
import quickdt.predictiveModels.decisionTree.tree.Leaf;
import quickdt.predictiveModels.decisionTree.tree.Node;

/**
 * Created with IntelliJ IDEA. User: janie Date: 6/26/13 Time: 3:15 PM To change
 * this template use File | Settings | File Templates.
 */
public class Tree implements PredictiveModel {
	static final long serialVersionUID = 56394564395635672L;

	public final Node             node;
	private ClassificationCounter classificationCounter;

	protected Tree(Node tree) {
		this.node = tree;
		classificationCounter = node.getClassificationCounter();
	}

	@Override
	public double getProbability(Attributes attributes, Serializable classification) {
		Leaf leaf = node.getLeaf(attributes);
		return leaf.getProbability(classification);
	}

	public ClassificationCounter getClassificationCounter() {
		return classificationCounter;
	}

	public void pruneDeepestLeaves() {
		node.pruneDeepestLeaves();
	}

	@Override
	public Map<Serializable, Double> getProbabilitiesByClassification(Attributes attributes) {
		Leaf leaf = node.getLeaf(attributes);
		Map<Serializable, Double> probsByClassification = Maps.newHashMap();
		for (Serializable classification : leaf.getClassifications()) {
			probsByClassification.put(classification, leaf.getProbability(classification));
		}
		return probsByClassification;
	}

	@Override
	public void dump(PrintStream printStream) {
		node.dump(printStream);
	}

	@Override
	public Serializable getClassificationByMaxProb(Attributes attributes) {
		Leaf leaf = node.getLeaf(attributes);
		return leaf.getBestClassification();
	}

	@Override
	public boolean equals(final Object o) {
		if (this == o) {
			return true;
		}
		if (o == null || getClass() != o.getClass()) {
			return false;
		}

		final Tree tree = (Tree) o;
		return node.equals(tree.node);
	}

	public List<Leaf> getLeavesWithMajorityIn(Serializable target) {
		return getLeaves().stream().filter(leaf -> leaf.getMajorityClass().equals(target))
				.collect(Collectors.toList());
	}

	public List<Leaf> getLeaves() {
		return node.collectLeaves();
	}

	public Map<Serializable, Double> getRecall() {
		Map<Serializable, Double> metric = new HashMap<>();

		Set<Serializable> targets = getTargets();
		for (Serializable target : targets) {
			List<Leaf> targetLeaves = getLeavesWithMajorityIn(target);
			double truePositives = targetLeaves.stream().map(Leaf::getTruePositives)
					.mapToDouble(Double::doubleValue).sum();
			double falseNegatives = getLeaves().stream().filter(o -> !targetLeaves.contains(o))
					.map(leaf -> leaf.getCountForClass(target)).mapToDouble(Double::doubleValue)
					.sum();
			metric.put(target, truePositives / (truePositives + falseNegatives));
		}
		return metric;
	}

	public Map<Serializable, Double> getPrecision() {
		Map<Serializable, Double> metric = new HashMap<>();

		Set<Serializable> targets = getTargets();
		for (Serializable target : targets) {
			List<Leaf> targetLeaves = getLeavesWithMajorityIn(target);
			double truePositives = targetLeaves.stream().map(Leaf::getTruePositives)
					.mapToDouble(Double::doubleValue).sum();
			double falsePositives = targetLeaves.stream().map(Leaf::getFalsePositives)
					.mapToDouble(Double::doubleValue).sum();
			metric.put(target, truePositives / (truePositives + falsePositives));
		}
		return metric;
	}

	private Set<Serializable> getTargets() {
		return getLeaves().stream().map(l -> l.classificationCounts.getCounts().keySet())
				.flatMap(Collection::stream).collect(Collectors.toSet());
	}

	@Override
	public int hashCode() {
		return node.hashCode();
	}
}
