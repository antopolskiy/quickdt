package quickdt.predictiveModels.decisionTree;

import quickdt.predictiveModels.decisionTree.tree.ClassCounter;

class InOutCounts {
	private ClassCounter in;
	private ClassCounter out;

	public ClassCounter in() {
		return in;
	}

	public ClassCounter out() {
		return out;
	}

	public InOutCounts(ClassCounter in, ClassCounter out) {
		this.in = in;
		this.out = out;
	}

	public void moveOutToIn(ClassCounter classCounter) {
		out = out.subtract(classCounter);
		in = in.add(classCounter);
	}

	public void moveInToOut(ClassCounter classCounter) {
		in = in.subtract(classCounter);
		out = out.add(classCounter);
	}

	public boolean totalsAreOverThreshold(int value) {
		return in.getTotal() >= value || out.getTotal() >= value;
	}
}
