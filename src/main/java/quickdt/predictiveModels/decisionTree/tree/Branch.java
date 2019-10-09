package quickdt.predictiveModels.decisionTree.tree;

import java.io.PrintStream;
import java.util.LinkedList;
import java.util.List;

import com.google.common.base.Predicate;

import quickdt.data.AbstractInstance;
import quickdt.data.Attributes;

public abstract class Branch extends Node {
	private static final long serialVersionUID = 8290012786245422175L;

	public final String attribute;

	public Node trueChild, falseChild;

	public Branch(Branch parent, final String attribute) {
		super(parent);
		this.attribute = attribute;
	}

	public abstract boolean decide(Attributes attributes);

	@Override
	public int size() {
		return 1 + trueChild.size() + falseChild.size();
	}

	public Predicate<AbstractInstance> getInPredicate() {
		return new Predicate<AbstractInstance>() {

			@Override
			public boolean apply(final AbstractInstance input) {
				return decide(input.getAttributes());
			}
		};
	}

	public Predicate<AbstractInstance> getOutPredicate() {
		return new Predicate<AbstractInstance>() {

			@Override
			public boolean apply(final AbstractInstance input) {
				return !decide(input.getAttributes());
			}
		};
	}

	@Override
	public Leaf getLeaf(final Attributes attributes) {
		if (decide(attributes)) {
			return trueChild.getLeaf(attributes);
		} else {
			return falseChild.getLeaf(attributes);
		}
	}

	@Override
	public void dump(final int indent, final PrintStream ps) {
		for (int x = 0; x < indent; x++) {
			ps.print(' ');
		}
		ps.println(this);
		trueChild.dump(indent + 2, ps);
		for (int x = 0; x < indent; x++) {
			ps.print(' ');
		}
		ps.println(toNotString());
		falseChild.dump(indent + 2, ps);
	}

	@Override
	public List<Leaf> collectLeaves() {
		List<Leaf> result = new LinkedList<>(trueChild.collectLeaves());
		result.addAll(falseChild.collectLeaves());
		return result;
	}

	@Override
	protected Leaf collapse(int newDepth) {
		Leaf newLeaf = new Leaf(parent, getClassificationCounter(), newDepth);

		if (!isRoot()) {
			if (parent.trueChild.equals(this)) {
				parent.trueChild = newLeaf;
			} else {
				parent.falseChild = newLeaf;
			}
		}
		return newLeaf;
	}

	@Override
	ClassificationCounter getClassificationCounter() {
		return ClassificationCounter.merge(trueChild.getClassificationCounter(),
				falseChild.getClassificationCounter());
	}

	public abstract String toNotString();

	@Override
	protected void calcMeanDepth(final LeafDepthStats stats) {
		trueChild.calcMeanDepth(stats);
		falseChild.calcMeanDepth(stats);
	}

	@Override
	public boolean equals(final Object o) {
		if (this == o) {
			return true;
		}
		if (o == null || getClass() != o.getClass()) {
			return false;
		}

		final Branch branch = (Branch) o;

		if (!attribute.equals(branch.attribute)) {
			return false;
		}
		if (!falseChild.equals(branch.falseChild)) {
			return false;
		}
		if (!trueChild.equals(branch.trueChild)) {
			return false;
		}

		return true;
	}

	@Override
	public int hashCode() {
		int result = attribute.hashCode();
		result = 31 * result + trueChild.hashCode();
		result = 31 * result + falseChild.hashCode();
		return result;
	}
}
