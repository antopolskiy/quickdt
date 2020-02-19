package quickdt.predictiveModels.decisionTree.tree;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.StreamSupport;

import org.apache.commons.lang.mutable.MutableInt;

import quickdt.data.AbstractInstance;

/**
 * IdAttibuteHandler is used to work with idAttributes. idAttribute is the name
 * of the column which has special function in the model. idAttribute is not
 * used for classification, i.e. it is exempt from the splits. However, it is
 * used to create special unique ID counts (see
 * {@link #countUniqueValues(int, Iterable, Map) countUniqueValues}: how many
 * data points with unique IDs are present in the leafs.
 */
public class IdAttributeHandler {
	/**
	 * Map containing ID attribute counts for each leaf, referenced by their hash.
	 */
	private Map<Integer, Map<Serializable, Integer>> countsMap = new HashMap<>();
	private Map<Serializable, Integer>               totalCounts;
	public final String                              idAttribute;

	public IdAttributeHandler() {
		this.idAttribute = null;
	}

	public IdAttributeHandler(String idAttribute) {
		this.idAttribute = idAttribute;
	}

	/**
	 * Count unique values of the idAttribute for each of the classes.
	 *
	 * Assumes that the first call will be made in the root node, and will save
	 * these stats separately in the {@link #totalCounts totalCounts}.
	 */
	public void countUniqueValues(int leafHash, Iterable<? extends AbstractInstance> trainingData,
			Map<Serializable, MutableInt> classifications) {

		Map<Serializable, Integer> idAttributeCounts = new HashMap<>();
		for (Map.Entry<Serializable, MutableInt> classification : classifications.entrySet()) {
			int classCount = Math.toIntExact(StreamSupport.stream(trainingData.spliterator(), false)
					.filter(s -> s.getClassification().equals(classification.getKey()))
					.map(s -> s.getAttributes().get(idAttribute)).distinct().count());
			idAttributeCounts.put(classification.getKey(), classCount);
		}

		// first count is the count of the root node, and is saved separately
		if (totalCounts == null) {
			totalCounts = idAttributeCounts;
		}

		countsMap.put(leafHash, idAttributeCounts);
	}

	public Map<Integer, Map<Serializable, Integer>> getCountsMap() {
		return countsMap;
	}

	public Map<Serializable, Integer> getCountForLeaf(int leafHash) {
		return getCountsMap().get(leafHash);
	}

	public Integer getCountForLeafClass(int leafHash, Serializable classification) {
		return getCountForLeaf(leafHash).get(classification);
	}

	public Integer getCountForMajorityClass(Leaf leaf) {
		return getCountForLeafClass(leaf.hashCode(), leaf.getBestClassification());
	}

	public Map<Serializable, Integer> getTotalCounts() {
		return totalCounts;
	}
}
