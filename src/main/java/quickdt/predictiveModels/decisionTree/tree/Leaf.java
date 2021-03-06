package quickdt.predictiveModels.decisionTree.tree;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;

import quickdt.data.AbstractInstance;
import quickdt.data.Attributes;

public class Leaf extends Node {
	private static final long serialVersionUID = -5617660873196498754L;

	private static final AtomicLong guidCounter = new AtomicLong(0);

	public final long guid;

	/**
	 * How deep in the tree is this label? A lower number typically indicates a more
	 * confident getBestClassification.
	 */
	public final int depth;

	/**
	 * How many training examples matched this leaf? A higher number indicates a
	 * more confident getBestClassification.
	 */
	public double exampleCount;

	/**
	 * The actual getBestClassification counts
	 */
	public final ClassCounter classificationCounts;

	public Leaf(Branch parent, final Iterable<? extends AbstractInstance> instances,
			final int depth) {
		this(parent, ClassCounter.countAll(instances), depth);
		Preconditions.checkArgument(!Iterables.isEmpty(instances),
				"Can't create leaf with no instances");
	}

	public Leaf(Branch parent, final ClassCounter classificationCounts, final int depth) {
		super(parent);
		guid = guidCounter.incrementAndGet();
		this.classificationCounts = classificationCounts;
		Preconditions.checkState(classificationCounts.getTotal() > 0,
				"Classifications must be > 0");
		exampleCount = classificationCounts.getTotal();
		this.depth = depth;
	}

	@Override
	public ClassCounter getClassificationCounter() {
		return classificationCounts;
	}

	@Override
	public void dump(final int indent, final PrintStream ps) {
		for (int x = 0; x < indent; x++) {
			ps.print(' ');
		}
		ps.println(this);
	}

	@Override
	public Leaf getLeaf(final Attributes attributes) {
		return this;
	}

	@Override
	public List<Leaf> collectLeaves() {
		return Collections.singletonList(this);
	}

	@Override
	public List<Node> collectNodes() {
		return Collections.singletonList(this);
	}

	@Override
	public int size() {
		return 1;
	}

	@Override
	protected void calcMeanDepth(final LeafDepthStats stats) {
		stats.ttlDepth += depth * exampleCount;
		stats.ttlSamples += exampleCount;
	}

	@Override
	protected Leaf collapse(int newDepth) {
		return this;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		Node currentNode = this;
		for (Branch n = parent; n != null; n = n.parent) {
			if (currentNode.equals(n.trueChild)) {
				builder.append(n.toString() + "->");
			} else {
				builder.append(n.toNotString() + "->");
			}
			currentNode = n;
		}
		builder.append("\n");
		for (Serializable key : getClassifications()) {
			builder.append(key + "=" + this.getProbability(key) + " ");
			builder.append("(matches=" + this.getTruePositives() + "; ");
			builder.append("contaminations=" + this.getFalsePositives() + ")");
			builder.append("\n");
		}
		return builder.toString();
	}

	public double getCountForClass(Serializable classification) {
		return getClassificationCounter().getCount(classification);
	}

	public Set<Serializable> getClassifications() {
		return getClassificationCounter().getCounts().keySet();
	}

	@Override
	public boolean equals(final Object o) {
		if (this == o) {
			return true;
		}
		if (o == null || getClass() != o.getClass()) {
			return false;
		}

		final Leaf leaf = (Leaf) o;

		if (depth != leaf.depth) {
			return false;
		}
		if (Double.compare(leaf.exampleCount, exampleCount) != 0) {
			return false;
		}
		if (!getClassificationCounter().equals(leaf.getClassificationCounter())) {
			return false;
		}

		return true;
	}

	@Override
	public int hashCode() {
		return new HashCodeBuilder(17, 37).append(depth).append(exampleCount).append(guidCounter)
				.append(classificationCounts).toHashCode();
	}

	Node getSibling() {
		if (isTrueChild()) {
			return parent.falseChild;
		} else if (isFalseChild()) {
			return parent.trueChild;
		} else if (isRoot()) {
			return null;
		} else {
			throw new NotImplementedException();
		}
	}

	private boolean isFalseChild() {
		if (!isRoot()) {
			return this == parent.falseChild;
		} else {
			return false;
		}
	}

	private boolean isTrueChild() {
		if (!isRoot()) {
			return this == parent.trueChild;
		} else {
			return false;
		}
	}

	Leaf pruneMe() {
		return parent.collapse(depth - 1);
	}

}
