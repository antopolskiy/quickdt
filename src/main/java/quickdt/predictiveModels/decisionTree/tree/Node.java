package quickdt.predictiveModels.decisionTree.tree;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import quickdt.data.Attributes;

public abstract class Node implements Serializable {
	private static final long serialVersionUID = -8713974861744567620L;

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
	public List<Leaf> pruneDeepestLeaves() {
		int depth = 0;
		List<Leaf> leaves = this.collectLeaves();
		List<Leaf> maxDepthLeaves = new ArrayList<>(leaves.size());
		for (Leaf leaf : leaves) {
			if (leaf.depth > depth) {
				depth = leaf.depth;
				maxDepthLeaves = new ArrayList<>(leaves.size());
			}
			if (leaf.depth == depth) {
				maxDepthLeaves.add(leaf);
			}
		}
		leaves = new ArrayList<>(leaves.size());
		for (Leaf leaf : maxDepthLeaves) {
			leaves.add(leaf.parent.collapse(depth - 1));
		}
		return leaves;
	}
}
