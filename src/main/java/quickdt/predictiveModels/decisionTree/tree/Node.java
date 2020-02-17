package quickdt.predictiveModels.decisionTree.tree;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import quickdt.data.Attributes;

public abstract class Node implements Serializable {
	private static final long serialVersionUID = -8713974861744567620L;

	protected transient volatile Map.Entry<Serializable, Double> bestClassificationEntry = null;

	protected synchronized Map.Entry<Serializable, Double> getBestClassificationEntry() {
		if (bestClassificationEntry != null) {
			return bestClassificationEntry;
		}

		for (Map.Entry<Serializable, Double> e : getClassificationCounter().getCounts()
				.entrySet()) {
			if (bestClassificationEntry == null
					|| e.getValue() > bestClassificationEntry.getValue()) {
				bestClassificationEntry = e;
			}
		}

		return bestClassificationEntry;
	}

	public abstract void dump(int indent, PrintStream ps);

	public final Branch parent;

	public Node(Branch parent) {
		this.parent = parent;
	}

	public boolean isRoot() {
		return parent == null;
	}

	/**
	 * Writes a textual representation of this tree to a PrintStream
	 *
	 * @param ps
	 */
	public void dump(final PrintStream ps) {
		dump(0, ps);
	}

	/**
	 * Get a label for a given set of HashMapAttributes
	 *
	 * @param attributes
	 * @return
	 */
	public abstract Leaf getLeaf(Attributes attributes);

	public abstract List<Leaf> collectLeaves();

	public abstract List<Node> collectNodes();

	/**
	 * Return the mean depth of leaves in the tree. A lower number generally
	 * indicates that the decision tree learner has done a better job.
	 *
	 * @return
	 */
	public double meanDepth() {
		final LeafDepthStats stats = new LeafDepthStats();
		calcMeanDepth(stats);
		return (double) stats.ttlDepth / stats.ttlSamples;
	}

	/**
	 * Return the number of nodes in this decision tree.
	 *
	 * @return
	 */
	public abstract int size();

	public abstract ClassificationCounter getClassificationCounter();

	/**
	 *
	 * @return The most likely classification
	 */
	public Serializable getBestClassification() {
		return getBestClassificationEntry().getKey();
	}

	public double getTruePositives() {
		return getClassificationCounter().getCount(getBestClassification());
	}

	public double getFalseNegatives(ClassificationCounter treeClassificationCounter) {
		return treeClassificationCounter.getCount(getBestClassification()) - getTruePositives();
	}

	public double getFalsePositives() {
		return getClassificationCounter().getTotal() - getTruePositives();
	}

	public Map<Serializable, Double> getFalsePositivesDistribution() {
		Serializable majority = getBestClassification();
		return getClassificationCounter().allClassifications().stream()
				.filter(cls -> !cls.equals(majority)).collect(Collectors.toMap(Function.identity(),
						c -> getClassificationCounter().getCount(c)));
	}

	public double getProbability(Serializable classification) {
		final double totalCount = getClassificationCounter().getTotal();
		if (totalCount == 0) {
			throw new IllegalStateException(
					"Trying to get a probability from a Leaf with no examples");
		}
		final double probability = getClassificationCounter().getCount(classification) / totalCount;
		return probability;
	}

	@Override
	public abstract boolean equals(final Object obj);

	@Override
	public abstract int hashCode();

	protected abstract void calcMeanDepth(LeafDepthStats stats);

//	public abstract void reduceDepth();

	protected abstract Leaf collapse(int newDepth);

	protected static class LeafDepthStats {
		int ttlDepth   = 0;
		int ttlSamples = 0;
	}

	/**
	 * @return the newly generated leaves, notice that these are not necessarily all
	 *         the leaves of the new highest depth
	 */
	Node collapseDeepestLeaves() {
		int depth = 0;
		List<Leaf> leaves = this.collectLeaves();

		List<Leaf> maxDepthLeaves = new ArrayList<>(leaves.size());
		for (Leaf leaf : leaves) {
			if (leaf.depth > depth) {
				depth = leaf.depth;
				maxDepthLeaves = new ArrayList<>(leaves.size());
			} else if (leaf.depth == depth) {
				maxDepthLeaves.add(leaf);
			}
		}

		for (Leaf leaf : maxDepthLeaves) {
			if (!leaf.isRoot()) {
				if (leaf.parent.isRoot()) {
					return leaf.parent.collapse(0);
				}
				leaf.parent.collapse(depth - 1);
			}
		}
		return this;
	}
}
