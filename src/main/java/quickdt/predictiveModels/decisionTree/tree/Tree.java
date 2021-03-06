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

	public final Node    node;
	private ClassCounter classCounter;

	public Tree(Node tree) {
		this.node = tree;
		classCounter = node.getClassificationCounter();
	}

	@Override
	public double getProbability(Attributes attributes, Serializable classification) {
		Leaf leaf = node.getLeaf(attributes);
		return leaf.getProbability(classification);
	}

	public ClassCounter getClassCounter() {
		return classCounter;
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

	public List<Node> getNodesWithMajorityIn(Serializable target) {
		return getNodes().stream().filter(leaf -> leaf.getBestClassification().equals(target))
				.collect(Collectors.toList());
	}

	public List<Leaf> getLeaves() {
		return node.collectLeaves();
	}

	public List<Node> getNodes() {
		return node.collectNodes();
	}

	private Set<Serializable> getTargets() {
		return classCounter.allClassifications();
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

	public Map<Serializable, Map<Serializable, Double>> getFalsePositiveDistribution() {
		Map<Serializable, Map<Serializable, Double>> dist = new HashMap<>();

		for (Serializable target : getTargets()) {
			List<Leaf> targetLeaves = getLeavesWithMajorityIn(target);
			Map<Serializable, Double> falsePositives = new HashMap<>();
			targetLeaves.forEach(l -> {
				Map<Serializable, Double> next = l.getFalsePositivesDistribution();
				next.forEach((k, v) -> {
					falsePositives.merge(k, v, Double::sum);
				});
			});
			dist.put(target, falsePositives);
		}
		return dist;
	}

	public Map<Serializable, Map<Serializable, Double>> getFalseNegativeDistribution() {
		Map<Serializable, Map<Serializable, Double>> dist = new HashMap<>();

		Set<Serializable> targets = getTargets();
		for (Serializable target : targets) {
			List<Leaf> targetLeaves = getLeavesWithMajorityIn(target);
			Map<Serializable, Double> falseNegatives = getLeaves().stream()
					.filter(o -> !targetLeaves.contains(o))
					.collect(Collectors.toMap(l -> l.getBestClassification(),
							l -> l.getCountForClass(target), Double::sum));
			dist.put(target, falseNegatives);
		}
		return dist;
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

	public Map<Serializable, Double> getF1() {
		Map<Serializable, Double> metric = new HashMap<>();
		Map<Serializable, Double> recall = getRecall();
		Map<Serializable, Double> precision = getPrecision();

		Set<Serializable> targets = getTargets();
		for (Serializable target : targets) {
			Double targetRecall = recall.get(target);
			Double targetPrecision = precision.get(target);
			double f1Score = 2 * (targetRecall * targetPrecision)
					/ (targetRecall + targetPrecision);
			metric.put(target, Double.isNaN(f1Score) ? 0.0 : f1Score);
		}
		return metric;
	}

	@Override
	public int hashCode() {
		return node.hashCode();
	}
}
