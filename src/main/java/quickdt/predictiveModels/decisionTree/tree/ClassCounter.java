package quickdt.predictiveModels.decisionTree.tree;

import static quickdt.predictiveModels.decisionTree.TreeBuilder.MISSING_VALUE;

import java.io.Serializable;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.javatuples.Pair;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import quickdt.collections.ValueSummingMap;
import quickdt.data.AbstractInstance;

public class ClassCounter implements Serializable {
	private static final long                   serialVersionUID = -6821237234748044623L;
	private final ValueSummingMap<Serializable> counts           = new ValueSummingMap<Serializable>();

	public static ClassCounter merge(ClassCounter a, ClassCounter b) {
		ClassCounter newCC = new ClassCounter();
		newCC.counts.putAll(a.counts);
		for (Entry<Serializable, Number> e : b.counts.entrySet()) {
			newCC.counts.addToValue(e.getKey(), e.getValue().doubleValue());
		}
		return newCC;
	}

	public static Pair<ClassCounter, Map<Serializable, ClassCounter>> countAllByAttributeValues(
			final Iterable<? extends AbstractInstance> instances, final String attribute,
			String splitAttribute, Serializable splitAttributeValue) {
		final Map<Serializable, ClassCounter> result = Maps.newHashMap();
		final ClassCounter totals = new ClassCounter();
		for (final AbstractInstance instance : instances) {
			final Serializable attrVal = instance.getAttributes().get(attribute);
			ClassCounter cc = null;
			boolean acceptableMissingValue = attrVal == null
					&& isAnAcceptableMissingValue(instance, splitAttribute, splitAttributeValue);

			if (attrVal != null) {
				cc = result.get(attrVal);
			} else if (acceptableMissingValue) {
				cc = result.get(MISSING_VALUE);
			} else {
				// ignore missing values
				continue;
			}

			if (cc == null) {
				cc = new ClassCounter();
				Serializable newKey = (attrVal != null) ? attrVal : MISSING_VALUE;
				result.put(newKey, cc);
			}
			cc.addClassification(instance.getClassification(), instance.getWeight());
			totals.addClassification(instance.getClassification(), instance.getWeight());
		}

		return Pair.with(totals, result);
	}

	/**
	 * Attributes with more skewed target counts come out first
	 */
	public static Pair<ClassCounter, List<AttrValClassCounter>> getSortedListOfAttrValuesWithClassCounters(
			final Iterable<? extends AbstractInstance> instances, final String attribute,
			String splitAttribute, Serializable splitAttributeValue,
			final Serializable minorityClassification) {

		final Pair<ClassCounter, Map<Serializable, ClassCounter>> pair = countAllByAttributeValues(
				instances, attribute, splitAttribute, splitAttributeValue);
		final ClassCounter totalCounter = pair.getValue0();
		final Map<Serializable, ClassCounter> attrValCounters = pair.getValue1();

		List<AttrValClassCounter> attrWithClassCounters = toSortedListOfValueClassCounters(
				attrValCounters, new MinorityProportionAndSizeComparator(minorityClassification));

		return Pair.with(totalCounter, attrWithClassCounters);
	}

	/**
	 * Given a map of attribute values to classification counters, and a comparator,
	 * produces a sorted list of AttrValueClassCounters. They contain the same
	 * information, packaged.
	 * 
	 * This sorted list can be used later by categorical branching algorithm to pick
	 * categorical values for optimal splits.
	 * 
	 * @param attrValCounters Map attribute value -> Classification Counter for that
	 *                        value
	 * @param comparator      Used to sort the list. E.g. see
	 *                        {@link MinorityProportionAndSizeComparator
	 *                        MinorityProportionAndSizeComparator}
	 * @return
	 */
	private static List<AttrValClassCounter> toSortedListOfValueClassCounters(
			Map<Serializable, ClassCounter> attrValCounters, Comparator comparator) {
		List<AttrValClassCounter> attributesWithClassificationCounters = Lists.newArrayList();
		for (Entry<Serializable, ClassCounter> entry : attrValCounters.entrySet()) {
			attributesWithClassificationCounters
					.add(new AttrValClassCounter(entry.getKey(), entry.getValue()));
		}
		attributesWithClassificationCounters.sort(comparator);
		return attributesWithClassificationCounters;
	}

	private static boolean isAnAcceptableMissingValue(AbstractInstance instance,
			String splitAttribute, Serializable splitAttributeValue) {
		return splitAttribute == null || splitAttributeValue == null
				|| instance.getAttributes().get(splitAttribute).equals(splitAttributeValue);
	}

	public Map<Serializable, Double> getCounts() {
		Map<Serializable, Double> ret = Maps.newHashMap();
		for (Entry<Serializable, Number> serializableNumberEntry : counts.entrySet()) {
			ret.put(serializableNumberEntry.getKey(),
					serializableNumberEntry.getValue().doubleValue());
		}
		return ret;
	}

	public static ClassCounter countAll(final Iterable<? extends AbstractInstance> instances) {
		final ClassCounter result = new ClassCounter();
		for (final AbstractInstance instance : instances) {
			result.addClassification(instance.getClassification(), instance.getWeight());
		}
		return result;
	}

	public void addClassification(final Serializable classification, double weight) {
		counts.addToValue(classification, weight);
	}

	public double getCount(final Serializable classification) {
		Number count = counts.get(classification);
		if (count == null) {
			return 0;
		} else {
			return count.doubleValue();
		}
	}

	public Set<Serializable> allClassifications() {
		return counts.keySet();
	}

	public ClassCounter add(final ClassCounter other) {
		final ClassCounter result = new ClassCounter();
		result.counts.putAll(counts);
		for (final Entry<Serializable, Number> e : other.counts.entrySet()) {
			result.counts.addToValue(e.getKey(), e.getValue().doubleValue());
		}
		return result;
	}

	public ClassCounter subtract(final ClassCounter other) {
		final ClassCounter result = new ClassCounter();
		result.counts.putAll(counts);
		for (final Entry<Serializable, Number> e : other.counts.entrySet()) {
			result.counts.addToValue(e.getKey(), -other.getCount(e.getKey()));
		}
		return result;
	}

	public double getTotal() {
		return counts.getSumOfValues();
	}

	public Pair<Serializable, Double> mostPopular() {
		Entry<Serializable, Number> best = null;
		for (final Entry<Serializable, Number> e : counts.entrySet()) {
			if (best == null || e.getValue().doubleValue() > best.getValue().doubleValue()) {
				best = e;
			}
		}
		return Pair.with(best.getKey(), best.getValue().doubleValue());
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) {
			return true;
		}
		if (o == null || getClass() != o.getClass()) {
			return false;
		}

		ClassCounter that = (ClassCounter) o;

		if (!counts.equals(that.counts)) {
			return false;
		}

		return true;
	}

	@Override
	public int hashCode() {
		return counts.hashCode();
	}

	private static class MinorityProportionComparator implements Comparator<AttrValClassCounter> {

		Serializable minorityClassification;

		public MinorityProportionComparator(Serializable minorityClassification) {
			this.minorityClassification = minorityClassification;
		}

		@Override
		public int compare(AttrValClassCounter cc1, AttrValClassCounter cc2) {
			double p1 = cc1.classCounter.getCount(minorityClassification)
					/ cc1.classCounter.getTotal();
			double p2 = cc2.classCounter.getCount(minorityClassification)
					/ cc2.classCounter.getTotal();

			return (int) Math.signum(p2 - p1);
		}
	}

	private static class MinorityProportionAndSizeComparator
			implements Comparator<AttrValClassCounter> {

		Serializable minorityClassification;

		public MinorityProportionAndSizeComparator(Serializable minorityClassification) {
			this.minorityClassification = minorityClassification;
		}

		@Override
		public int compare(AttrValClassCounter cc1, AttrValClassCounter cc2) {
			double cc1Total = cc1.classCounter.getTotal();
			double p1 = cc1.classCounter.getCount(minorityClassification) / cc1Total;
			double cc2Total = cc2.classCounter.getTotal();
			double p2 = cc2.classCounter.getCount(minorityClassification) / cc2Total;

			int signum = (int) Math.signum(p2 - p1);
			return signum == 0 ? (int) Math.signum(cc2Total - cc1Total) : signum;
		}
	}
}
