package quickdt.predictiveModels.decisionTree.tree;

import java.io.Serializable;

/**
 * Contains attribute value and the classification counter associated with this
 * value.
 */
// TODO Make this a map?
public class AttrValClassCounter {
	public Serializable attrValue;
	public ClassCounter classCounter;

	public AttrValClassCounter(Serializable attrValue, ClassCounter classCounter) {
		this.attrValue = attrValue;
		this.classCounter = classCounter;
	}
}
