package quickdt.predictiveModels.decisionTree.tree;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.collect.Maps;

import quickdt.data.Attributes;
import quickdt.predictiveModels.PredictiveModel;

/**
 * Created with IntelliJ IDEA. User: janie Date: 6/26/13 Time: 3:15 PM To change
 * this template use File | Settings | File Templates.
 */
public class Tree implements PredictiveModel {
	static final long serialVersionUID = 56394564395635672L;

	public final Node             node;
	private ClassificationCounter classificationCounter;

	public Tree(Node tree) {
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

	/**
	 * @param prune after collapsing deepest leaves, prune leaves with the same
	 *              majority category.
	 */
	public Tree collapseDeepestLeaves(boolean prune) {
		Tree tree = new Tree(node.collapseDeepestLeaves());
		if (prune) {
			tree = tree.pruneSameCategoryLeaves();
		}
		return tree;
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
		return getLeaves().stream().filter(leaf -> leaf.getBestClassification().equals(target))
				.collect(Collectors.toList());
	}

	public List<Leaf> getLeaves() {
		return node.collectLeaves();
	}

	private Set<Serializable> getTargets() {
		return classificationCounter.allClassifications();
	}

	public int getMaxDepth() {
		Optional<Leaf> leaf = getLeaves().stream().max(Comparator.comparingInt(x -> x.depth));
		return leaf.map(value -> value.depth).orElse(0);
	}

	public Map<Serializable, Double> getRecall() {
		Map<Serializable, Double> metric = new HashMap<>();
		Map<Serializable, Double> truePositiveCounts = getTruePositiveCounts();
		Map<Serializable, Double> falseNegativesCounts = getFalseNegativesCounts();

		Set<Serializable> targets = getTargets();
		for (Serializable target : targets) {
			Double truePositives = truePositiveCounts.get(target);
			if (truePositives == null) {
				metric.put(target, 0.0);
			} else {
				Double falseNegatives = falseNegativesCounts.getOrDefault(target, 0.0);
				metric.put(target, truePositives / (truePositives + falseNegatives));
			}
		}
		return metric;
	}

	public Map<Serializable, Double> getPrecision() {
		Map<Serializable, Double> metric = new HashMap<>();
		Map<Serializable, Double> truePositiveCounts = getTruePositiveCounts();
		Map<Serializable, Double> falsePositiveCounts = getFalsePositiveCounts();

		Set<Serializable> targets = getTargets();
		for (Serializable target : targets) {
			Double truePositives = truePositiveCounts.get(target);
			if (truePositives == null) {
				metric.put(target, 0.0);
			} else {
				Double falsePositives = falsePositiveCounts.getOrDefault(target, 0.0);
				metric.put(target, truePositives / (truePositives + falsePositives));
			}
		}
		return metric;
	}

	public Map<Serializable, Double> getTruePositiveCounts() {
		Map<Serializable, Double> counts = new HashMap<>();

		Set<Serializable> targets = getTargets();
		for (Serializable target : targets) {
			List<Leaf> targetLeaves = getLeavesWithMajorityIn(target);
			double truePositives = targetLeaves.stream().map(Leaf::getTruePositives)
					.mapToDouble(Double::doubleValue).sum();
			counts.put(target, truePositives);
		}
		return counts;
	}

	public Map<Serializable, Double> getFalsePositiveCounts() {
		Map<Serializable, Double> counts = new HashMap<>();

		Set<Serializable> targets = getTargets();
		for (Serializable target : targets) {
			List<Leaf> targetLeaves = getLeavesWithMajorityIn(target);
			double falsePositives = targetLeaves.stream().map(Leaf::getFalsePositives)
					.mapToDouble(Double::doubleValue).sum();
			counts.put(target, falsePositives);
		}
		return counts;
	}

	public Map<Serializable, Double> getFalseNegativesCounts() {
		Map<Serializable, Double> counts = new HashMap<>();

		Set<Serializable> targets = getTargets();
		for (Serializable target : targets) {
			List<Leaf> targetLeaves = getLeavesWithMajorityIn(target);
			double falseNegatives = getLeaves().stream().filter(o -> !targetLeaves.contains(o))
					.map(leaf -> leaf.getCountForClass(target)).mapToDouble(Double::doubleValue)
					.sum();
			counts.put(target, falseNegatives);
		}
		return counts;
	}

	public Tree pruneSameCategoryLeaves() {
		boolean notFinished = true;
		while (notFinished) {
			notFinished = false;
			List<Leaf> leaves = getLeaves();
			for (Leaf leaf : leaves) {
				if (leaf.getSibling() instanceof Leaf) {
					if (leaf.getBestClassification()
							.equals(((Leaf) leaf.getSibling()).getBestClassification())) {
						Leaf collapsedLeaf = leaf.pruneMe();
						if (collapsedLeaf.isRoot()) {
							return new Tree(collapsedLeaf);
						}
						notFinished = true;
						break;
					}
				}
			}
		}
		return this;
	}

	@Override
	public int hashCode() {
		return node.hashCode();
	}
}
