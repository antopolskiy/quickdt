package quickdt.predictiveModels.decisionTree;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;

import org.javatuples.Pair;

import quickdt.predictiveModels.decisionTree.tree.AttrValClassCounter;
import quickdt.predictiveModels.decisionTree.tree.ClassCounter;

/**
 * Handles ignored values; These are the values, which cannot be used in splits,
 * but their counts are still tracked.
 * 
 * The important concept to understand is that the samples associated with the
 * ignored values will always be in the FALSE branch of the split, as for any
 * possible split, they are always NOT [set of values to be split upon].
 */
public class IgnoredValuesHandler {
	private final ClassCounter       counter;
	private final List<Serializable> values;
	public final boolean             skip;
	private boolean                  flipped = false;

	public IgnoredValuesHandler(List<Serializable> ignoredValues,
			List<AttrValClassCounter> attrValClassCounters) {
		values = ignoredValues;
		if (values.isEmpty()) {
			counter = new ClassCounter();
			skip = true;
		} else {
			counter = getCount(attrValClassCounters);
			skip = false;
		}
	}

	public ClassCounter getCounter() {
		return counter;
	}

	private ClassCounter getCount(List<AttrValClassCounter> attrValClassCounters) {
		ClassCounter ignoredCounter = new ClassCounter();
		for (AttrValClassCounter c : attrValClassCounters) {
			if (values.contains(c.attrValue)) {
				ignoredCounter = ignoredCounter.add(c.classCounter);
			}
		}
		return ignoredCounter;
	}

	public Pair<ClassCounter, List<AttrValClassCounter>> removeValuesFromSplitCandidates(
			Pair<ClassCounter, List<AttrValClassCounter>> valueOutcomeCountsPairs) {

		// filter out the value to be ignored
		List<AttrValClassCounter> attrValClassCounters = valueOutcomeCountsPairs.getValue1()
				.stream().filter(c -> !values.contains(c.attrValue)).collect(Collectors.toList());

		return Pair.with(valueOutcomeCountsPairs.getValue0(), attrValClassCounters);
	}

	public void nowFlipped() {
		flipped = true;
	}

	public boolean isFlipped() {
		return flipped;
	}
}
