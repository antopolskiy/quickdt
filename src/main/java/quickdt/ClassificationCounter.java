package quickdt;

import com.google.common.collect.Maps;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleMaps;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import org.javatuples.Pair;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;

public class ClassificationCounter implements Serializable {
	private final Object2DoubleMap<Serializable> counts = new Object2DoubleOpenHashMap<Serializable>();

	private double total = 0;

	public static Pair<ClassificationCounter, Map<Serializable, ClassificationCounter>> countAllByAttributeValues(
			final Iterable<? extends AbstractInstance> instances, final String attribute) {
		final Map<Serializable, ClassificationCounter> result = Maps.newHashMap();
		final ClassificationCounter totals = new ClassificationCounter();
		for (final AbstractInstance i : instances) {
			final Serializable attrVal = i.getAttributes().get(attribute);
			if (attrVal != null) {
				ClassificationCounter cc = result.get(attrVal);
				if (cc == null) {
					cc = new ClassificationCounter();
					result.put(attrVal, cc);
				}
				cc.addClassification(i.getClassification(), i.getWeight());
				totals.addClassification(i.getClassification(), i.getWeight());
			}
		}
		return Pair.with(totals, result);
	}

    public Object2DoubleMap<Serializable> getCounts() {
        return Object2DoubleMaps.unmodifiable(counts);
    }


	public static ClassificationCounter countAll(final Iterable<? extends AbstractInstance> instances) {
		final ClassificationCounter result = new ClassificationCounter();
		for (final AbstractInstance i : instances) {
			result.addClassification(i.getClassification(), i.getWeight());
		}
		return result;
	}

	public void addClassification(final Serializable classification, double weight) {
		double c = counts.getDouble(classification); // should return 0.0 on absent values
		total+= weight;
		counts.put(classification, c + weight);
	}

	public double getCount(final Serializable classification) {
		return counts.getDouble(classification);
	}

	public Set<Serializable> allClassifications() {
		return counts.keySet();
	}

	public ClassificationCounter add(final ClassificationCounter other) {
		final ClassificationCounter result = new ClassificationCounter();
		result.counts.putAll(counts);
		for (final Object2DoubleMap.Entry<Serializable> e : other.counts.object2DoubleEntrySet()) {
			result.counts.put(e.getKey(), getCount(e.getKey()) + e.getDoubleValue());
		}
		result.total = total + other.total;
		return result;
	}

	public ClassificationCounter subtract(final ClassificationCounter other) {
		final ClassificationCounter result = new ClassificationCounter();
		for (final Object2DoubleMap.Entry<Serializable> e : counts.object2DoubleEntrySet()) {
			result.counts.put(e.getKey(), e.getDoubleValue() - other.getCount(e.getKey()));
		}
		result.total = total - other.total;
		return result;
	}

	public double getTotal() {
		return total;
	}

	public Pair<Serializable, Double> mostPopular() {
		Object2DoubleMap.Entry<Serializable> best = null;
		for (final Object2DoubleMap.Entry<Serializable> e : counts.object2DoubleEntrySet()) {
			if (best == null || e.getDoubleValue() > best.getDoubleValue()) {
				best = e;
			}
		}
		return Pair.with(best.getKey(), best.getValue());
	}
}
