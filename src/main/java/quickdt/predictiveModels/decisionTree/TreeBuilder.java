package quickdt.predictiveModels.decisionTree;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import javax.annotation.Nullable;

import org.apache.commons.lang.mutable.MutableInt;
import org.javatuples.Pair;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.twitter.common.stats.ReservoirSampler;

import quickdt.Misc;
import quickdt.data.AbstractInstance;
import quickdt.predictiveModels.UpdatablePredictiveModelBuilder;
import quickdt.predictiveModels.decisionTree.scorers.MSEScorer;
import quickdt.predictiveModels.decisionTree.tree.AttrValClassCounter;
import quickdt.predictiveModels.decisionTree.tree.Branch;
import quickdt.predictiveModels.decisionTree.tree.CategoricalBranch;
import quickdt.predictiveModels.decisionTree.tree.ClassCounter;
import quickdt.predictiveModels.decisionTree.tree.IdAttributeHandler;
import quickdt.predictiveModels.decisionTree.tree.Leaf;
import quickdt.predictiveModels.decisionTree.tree.Node;
import quickdt.predictiveModels.decisionTree.tree.NumericBranch;
import quickdt.predictiveModels.decisionTree.tree.Tree;
import quickdt.predictiveModels.decisionTree.tree.UpdatableLeaf;

public final class TreeBuilder implements UpdatablePredictiveModelBuilder<Tree> {
	private static final int         ORDINAL_TEST_SPLITS = 5;
	private static final int         RESERVOIR_SIZE      = 1000;
	public static final Serializable MISSING_VALUE       = "%missingVALUE%83257";

	private final Scorer                      scorer;
	private int                               maxDepth                           = Integer.MAX_VALUE;
	private int                               maxCategoricalInSetSize            = Integer.MAX_VALUE;
	private int                               minInstancesPerCategoricalVariable = 0;
	private int                               minLeafInstances                   = 0;
	private boolean                           updatable                          = false;
	private boolean                           forceSplitsOnMissing               = false;
	private boolean                           binaryClassifications              = true;
	private boolean                           pruneSameCategory                  = false;
	private boolean                           treatNumericAsCategorical          = false;
	private double                            minimumScore                       = 0.00000000000001;
	private double                            ignoreAttributeAtNodeProbability   = 0.0;
	private double                            eps                                = .000001;
	private Serializable                      minorityClassification;
	private String                            splitAttribute                     = null;
	private Set<String>                       splitModelWhiteList;
	private Serializable                      id;
	private IdAttributeHandler                idAttributeHandler                 = new IdAttributeHandler();
	private HashMap<Serializable, MutableInt> classifications                    = Maps
			.newHashMap();
	private List<Serializable>                ignoredValues                      = new ArrayList<>();

	public TreeBuilder() {
		this(new MSEScorer(MSEScorer.CrossValidationCorrection.FALSE));
	}

	public TreeBuilder(final Scorer scorer) {
		this.scorer = scorer;
	}

	/**
	 * In case of columns with null values, this controls whether a split on null
	 * values should be forced.
	 * 
	 * This is especially relevant for numeric columns with null values, because it
	 * allows the first split to be on null only, and then treating the remaining
	 * splits as proper numeric variable.
	 * 
	 * In case of categorical variables, if missing variables are represented by
	 * empty strings (""), this does not impact them. If missing values are
	 * represented by null, it does.
	 */
	public TreeBuilder forceSplitOnNull() {
		this.forceSplitsOnMissing = true;
		return this;
	}

	public TreeBuilder maxDepth(int maxDepth) {
		this.maxDepth = maxDepth;
		return this;
	}

	public TreeBuilder maxCategoricalInSetSize(int maxCategoricalInSetSize) {
		this.maxCategoricalInSetSize = maxCategoricalInSetSize;
		return this;
	}

	public TreeBuilder binaryClassification(boolean binaryClassification) {
		this.binaryClassifications = binaryClassification;
		return this;
	}

	public TreeBuilder minLeafInstances(int minLeafInstances) {
		this.minLeafInstances = minLeafInstances;
		return this;
	}

	public TreeBuilder splitPredictiveModel(String splitAttribute,
			Set<String> splitModelWhiteList) {
		this.splitAttribute = splitAttribute;
		this.splitModelWhiteList = splitModelWhiteList;
		return this;
	}

	public TreeBuilder ignoreAttributeAtNodeProbability(double probability) {
		this.ignoreAttributeAtNodeProbability = probability;
		return this;
	}

	public TreeBuilder minInstancesPerCategoricalVariable(int minLimit) {
		this.minInstancesPerCategoricalVariable = minLimit;
		return this;
	}

	public TreeBuilder minimumScore(double minimumScore) {
		this.minimumScore = minimumScore;
		return this;
	}

	/**
	 * After training the model, prune the leaves with the same majority category.
	 */
	public TreeBuilder pruneSameCategory() {
		this.pruneSameCategory = true;
		return this;
	}

	/**
	 * See {@link IdAttributeHandler} for detail.
	 */
	public TreeBuilder setIdAttribute(String idAttribute) {
		this.idAttributeHandler = new IdAttributeHandler(idAttribute);
		return this;
	}

	@Override
	public TreeBuilder updatable(boolean updatable) {
		this.updatable = updatable;
		return this;
	}

	public TreeBuilder ignoreEmptyStrings() {
		this.ignoredValues.add("");
		return this;
	}

	public TreeBuilder ignoreValue(Serializable value) {
		this.ignoredValues.add(value);
		return this;
	}

	public TreeBuilder ignoreValues(List<Serializable> values) {
		this.ignoredValues.addAll(values);
		return this;
	}

	public TreeBuilder treatNumericAsCategorical() {
		this.treatNumericAsCategorical = true;
		return this;
	}

	@Override
	public void setID(Serializable id) {
		this.id = id;
	}

	public IdAttributeHandler getIdAttributeHandler() {
		return idAttributeHandler;
	}

	@Override
	public Tree buildPredictiveModel(final Iterable<? extends AbstractInstance> trainingData) {
		setBinaryClassificationProperties(trainingData);
		Tree tree = new Tree(buildTree(null, trainingData, 0, createNumericSplits(trainingData)));
		if (pruneSameCategory) {
			return tree.pruneSameCategoryLeaves();
		}
		return tree;
	}

	@Override
	public void updatePredictiveModel(Tree tree, final Iterable<? extends AbstractInstance> newData,
			List<? extends AbstractInstance> trainingData, boolean splitNodes) {
		// first move all the data into the leaves
		for (AbstractInstance instance : newData) {
			addInstanceToNode(tree.node, instance);
		}
		// now split the leaves further if possible
		if (splitNodes) {
			splitNode(tree.node, trainingData);
		}
	}

	private void setBinaryClassificationProperties(
			Iterable<? extends AbstractInstance> trainingData) {

		for (AbstractInstance instance : trainingData) {
			Serializable classification = instance.getClassification();
			if (classifications.containsKey(classification)) {
				classifications.get(classification).increment();
			} else {
				classifications.put(classification, new MutableInt(1));
			}

			if (classifications.size() > 2) {
				binaryClassifications = false;
				return;
			}
		}

		minorityClassification = null;
		double minorityClassificationCount = 0;

		for (Entry<Serializable, MutableInt> entry : classifications.entrySet()) {
			if (minorityClassification == null
					|| entry.getValue().doubleValue() < minorityClassificationCount) {
				minorityClassification = entry.getKey();
				minorityClassificationCount = entry.getValue().doubleValue();
			}
		}
	}

	@Override
	public void stripData(Tree tree) {
		stripNode(tree.node);
	}

	private double[] createNumericSplit(final Iterable<? extends AbstractInstance> trainingData,
			final String attribute) {
		final ReservoirSampler<Double> reservoirSampler = new ReservoirSampler<>(RESERVOIR_SIZE);
		for (final AbstractInstance instance : trainingData) {
			Serializable value = instance.getAttributes().get(attribute);
			if (value == null) {
				value = 0;
			}
			reservoirSampler.sample(((Number) value).doubleValue());
		}

		return getSplit(reservoirSampler);
	}

	private Map<String, double[]> createNumericSplits(
			final Iterable<? extends AbstractInstance> trainingData) {
		final Map<String, ReservoirSampler<Double>> rsm = Maps.newHashMap();
		for (final AbstractInstance instance : trainingData) {
			for (final Entry<String, Serializable> attributeEntry : instance.getAttributes()
					.entrySet()) {
				if (attributeEntry.getValue() instanceof Number) {
					ReservoirSampler<Double> reservoirSampler = rsm.get(attributeEntry.getKey());
					if (reservoirSampler == null) {
						reservoirSampler = new ReservoirSampler<>(RESERVOIR_SIZE);
						rsm.put(attributeEntry.getKey(), reservoirSampler);
					}
					reservoirSampler.sample(((Number) attributeEntry.getValue()).doubleValue());
				}
			}
		}

		final Map<String, double[]> splits = Maps.newHashMap();

		for (final Entry<String, ReservoirSampler<Double>> e : rsm.entrySet()) {
			final double[] split = getSplit(e.getValue());
			splits.put(e.getKey(), split);
		}
		return splits;
	}

	private double[] getSplit(ReservoirSampler<Double> reservoirSampler) {
		final ArrayList<Double> splitList = Lists.newArrayList();
		for (final Double sample : reservoirSampler.getSamples()) {
			splitList.add(sample);
		}
		if (splitList.isEmpty()) {
			throw new RuntimeException("Split list empty");
		}
		Collections.sort(splitList);

		final double[] split = new double[ORDINAL_TEST_SPLITS - 1];
		final int indexMultiplier = splitList.size() / (split.length + 1);
		for (int x = 0; x < split.length; x++) {
			split[x] = splitList.get((x + 1) * indexMultiplier);
		}
		return split;
	}

	/**
	 * @param parent
	 * @param trainingData
	 * @param depth
	 * @param splits       Map argument -> numeric thresholds to be tested for
	 *                     splits
	 * @return
	 */
	private Node buildTree(Branch parent, final Iterable<? extends AbstractInstance> trainingData,
			final int depth, final Map<String, double[]> splits) {
		Preconditions.checkArgument(!Iterables.isEmpty(trainingData),
				"At Depth: " + depth + ". Can't build a tree with no training data");
		final Leaf thisLeaf;
		if (updatable) {
			thisLeaf = new UpdatableLeaf(parent, trainingData, depth);
		} else {
			thisLeaf = new Leaf(parent, trainingData, depth);
		}

		if (idAttributeHandler.idAttribute != null) {
			idAttributeHandler.countUniqueValues(thisLeaf, trainingData, classifications);
		}

		if (depth >= maxDepth) {
			return thisLeaf;
		}

		Pair<? extends Branch, Double> bestPair = getBestNodePair(parent, trainingData, splits);
		Branch bestNode = bestPair != null ? bestPair.getValue0() : null;
		double bestScore = bestPair != null ? bestPair.getValue1() : 0;

		// If we were unable to find a useful branch, return the leaf
		if (bestNode == null || bestScore < minimumScore) {
			// will be null if all attributes are ignored, and best score will be 0 if
			// 1 of 3 things happen: (1) all instances in the node have the same
			// classification, (2) each attribute tried has just 1 observed value
			// (3) subsets with the same attribute value have the same distribution of
			// classifications
			return thisLeaf;
		}

		final ArrayList<AbstractInstance> trueTrainingSet = Lists.newArrayList();
		final ArrayList<AbstractInstance> falseTrainingSet = Lists.newArrayList();
		setTrueAndFalseTrainingSets(trainingData, bestNode, trueTrainingSet, falseTrainingSet);

		if (trueTrainingSet.size() < this.minLeafInstances) {
			return thisLeaf;
		}

		if (falseTrainingSet.size() < this.minLeafInstances) {
			return thisLeaf;
		}

		double trueWeight = getTotalWeight(trueTrainingSet);
		double falseWeight = getTotalWeight(falseTrainingSet);
		if (trueWeight == 0 || falseWeight == 0) {
			return thisLeaf;
		}

		double[] oldSplit = null;
		// Temporarily replace the split for an attribute for
		// descendants of an numeric branch, first the trueTrainingSet split.
		// this is done to avoid copying the splits map at each iteration of the
		// recursive call. instead it is temporarily changed for the subtree, and then
		// recovered in the end to keep it consistent with the next calls, once the
		// subtree is processed.
		if (bestNode instanceof NumericBranch) {
			final NumericBranch bestBranch = (NumericBranch) bestNode;
			oldSplit = splits.get(bestBranch.attribute);
			splits.put(bestBranch.attribute,
					createNumericSplit(trueTrainingSet, bestBranch.attribute));
		}

		// Recurse down the true branch
		bestNode.trueChild = buildTree(bestNode, trueTrainingSet, depth + 1, splits);

		// Now the falseTrainingSet splits
		if (bestNode instanceof NumericBranch) {
			final NumericBranch bestBranch = (NumericBranch) bestNode;
			splits.put(bestBranch.attribute,
					createNumericSplit(falseTrainingSet, bestBranch.attribute));
		}

		// Recurse down the false branch
		bestNode.falseChild = buildTree(bestNode, falseTrainingSet, depth + 1, splits, false);

		// And now replace the original split if this is an NumericBranch
		// todo: this behavior can be extracted into NumericBranch method; other
		// branches would have empty method calls
		if (bestNode instanceof NumericBranch) {
			final NumericBranch bestBranch = (NumericBranch) bestNode;
			splits.put(bestBranch.attribute, oldSplit);
		}
		return bestNode;
	}

	private void setTrueAndFalseTrainingSets(Iterable<? extends AbstractInstance> trainingData,
			Branch bestNode, ArrayList<AbstractInstance> trueTrainingSet,
			ArrayList<AbstractInstance> falseTrainingSet) {
		final ArrayList<AbstractInstance> supportingDataSet = Lists.newArrayList();

		// put instances with attribute values into appropriate training sets
		for (AbstractInstance instance : trainingData) {
			boolean isASupportingInstanceFromADifferentSplit = false;
			boolean instanceNotPermittedToContributeToInsetDefinition = false;
			boolean usingSplitModel = splitAttribute != null && id != null;
			if (usingSplitModel) {
				isASupportingInstanceFromADifferentSplit = !instance.getAttributes()
						.get(splitAttribute).equals(id);
				instanceNotPermittedToContributeToInsetDefinition = !splitModelWhiteList
						.contains(bestNode.attribute);
			}

			boolean instanceIsInTheSupportingDataSet = usingSplitModel
					&& isASupportingInstanceFromADifferentSplit
					&& instanceNotPermittedToContributeToInsetDefinition; // and the attribute isn't
																			// in the whitelist
			if (instanceIsInTheSupportingDataSet) {
				supportingDataSet.add(instance);
			} else {
				if (bestNode.decide(instance.getAttributes())) {
					trueTrainingSet.add(instance);
				} else {
					falseTrainingSet.add(instance);
				}
			}
		}

		// put instances without values for the split attribute in the true and false
		// set in proper proportions.
		for (AbstractInstance instance : supportingDataSet) {
			double trueThreshold = trueTrainingSet.size()
					/ (trueTrainingSet.size() + falseTrainingSet.size());
			Random rand = Misc.random;
			if (rand.nextDouble() < trueThreshold) {
				trueTrainingSet.add(instance);
			} else {
				falseTrainingSet.add(instance);
			}
		}
	}

	private Pair<? extends Branch, Double> getBestNodePair(Branch parent,
			final Iterable<? extends AbstractInstance> trainingData,
			final Map<String, double[]> splits) {
		// should not be doing the following operation every time we call buildTree
		Map<String, AttributeCharacteristics> attributeCharacteristics = surveyTrainingData(
				trainingData);

		Pair<? extends Branch, Double> bestPair = null;

		for (final Entry<String, AttributeCharacteristics> attributeCharacteristicsEntry : attributeCharacteristics
				.entrySet()) {

			// skip ID attribute (see IdAttributeHandler for details)
			if (attributeCharacteristicsEntry.getKey()
					.equals(this.idAttributeHandler.idAttribute)) {
				continue;
			}

			if (this.ignoreAttributeAtNodeProbability > 0
					&& Misc.random.nextDouble() < this.ignoreAttributeAtNodeProbability) {// ||
																							// attributeCharacteristicsEntry.getKey().equals(splitAttribute))
																							// {
				continue;
			}

			Pair<? extends Branch, Double> thisPair = null;
			Pair<? extends Branch, Double> numericPair = null;
			Pair<? extends Branch, Double> categoricalPair = null;

			if (attributeCharacteristicsEntry.getValue().isNumber) {
				numericPair = createNumericNode(parent, attributeCharacteristicsEntry.getKey(),
						trainingData, splits.get(attributeCharacteristicsEntry.getKey()));
			} else {
				categoricalPair = createCategoricalNode(parent,
						attributeCharacteristicsEntry.getKey(), trainingData);
			}

			if (numericPair != null) {
				thisPair = numericPair;
			} else {
				// (numericPair.getValue1() > categoricalPair.getValue1()) ? numericPair :
				// categoricalPair;
				thisPair = categoricalPair;
			}
			if (bestPair == null
					|| (thisPair != null && thisPair.getValue1() > bestPair.getValue1())) {
				bestPair = thisPair;
			}
		}
		return bestPair;
	}

	private double getTotalWeight(List<? extends AbstractInstance> trainingSet) {
		double trueWeight = 0;
		for (AbstractInstance instance : trainingSet) {
			trueWeight += instance.getWeight();
		}
		return trueWeight;
	}

	/**
	 * Survey the data and determine for each attribute (column) if it is numeric
	 * (all instances are of class Number, no nulls) or categorical (all other
	 * cases).
	 * 
	 * @param trainingData
	 * @return
	 */
	private Map<String, AttributeCharacteristics> surveyTrainingData(
			final Iterable<? extends AbstractInstance> trainingData) {
		// tells us if each attribute is numeric or not.
		Map<String, AttributeCharacteristics> attributeCharacteristics = Maps.newHashMap();

		for (AbstractInstance instance : trainingData) {
			for (Entry<String, Serializable> e : instance.getAttributes().entrySet()) {
				AttributeCharacteristics attributeCharacteristic = attributeCharacteristics
						.get(e.getKey());
				if (attributeCharacteristic == null) {
					attributeCharacteristic = new AttributeCharacteristics();
					attributeCharacteristics.put(e.getKey(), attributeCharacteristic);
				}
				if (treatNumericAsCategorical || !(e.getValue() instanceof Number)) {
					attributeCharacteristic.isNumber = false;
				}
			}
		}
		return attributeCharacteristics;
	}

	private List<AttrValClassCounter> putMissingOnTop(
			List<AttrValClassCounter> valuesWithClassificationCounters) {
		List<AttrValClassCounter> counterForMissingList = getClassificationCountersForMissingValues(
				valuesWithClassificationCounters);
		AttrValClassCounter counterForMissingValues = counterForMissingList.get(0);
		valuesWithClassificationCounters.remove(counterForMissingValues);
		valuesWithClassificationCounters.add(0, counterForMissingValues);
		return valuesWithClassificationCounters;
	}

	private boolean hasMissing(List<AttrValClassCounter> valuesWithClassificationCounters) {
		List<AttrValClassCounter> counterForMissing = getClassificationCountersForMissingValues(
				valuesWithClassificationCounters);
		return !counterForMissing.isEmpty();
	}

	private List<AttrValClassCounter> getClassificationCountersForMissingValues(
			List<AttrValClassCounter> valuesWithClassificationCounters) {
		return valuesWithClassificationCounters.stream()
				.filter(a -> a.attrValue.equals(MISSING_VALUE)).collect(Collectors.toList());
	}

	private Pair<? extends Branch, Double> createCategoricalNode(Branch parent,
			final String attribute, final Iterable<? extends AbstractInstance> instances) {
		if (binaryClassifications) {
			return createTwoClassCategoricalNode(parent, attribute, instances);
		} else {
			return createNClassCategoricalNode(parent, attribute, instances);
		}
	}

	private Pair<? extends Branch, Double> createTwoClassCategoricalNode(Branch parent,
			final String attribute, final Iterable<? extends AbstractInstance> instances) {

		Pair<ClassCounter, List<AttrValClassCounter>> valueOutcomeCountsPairs = getSortedValueClassCounters(
				attribute, instances);

		// handle ignored values
		IgnoredValuesHandler ignoredValuesHandler = new IgnoredValuesHandler(this.ignoredValues,
				valueOutcomeCountsPairs.getValue1());
		valueOutcomeCountsPairs = ignoredValuesHandler
				.removeValuesFromSplitCandidates(valueOutcomeCountsPairs);

		final ClassCounter totalCounter = valueOutcomeCountsPairs.getValue0();
		final List<AttrValClassCounter> valuesWithClassCounters = valueOutcomeCountsPairs
				.getValue1();

		// classification counters for combining attributes
		InOutCounts inOutCounts = new InOutCounts(new ClassCounter(), totalCounter);

		ScoreTracker scoreTracker = new ScoreTracker(scorer, totalCounter);

		// force a split on missing
		if (forceSplitsOnMissing && hasMissing(valuesWithClassCounters)) {
			List<AttrValClassCounter> missingCounters = getClassificationCountersForMissingValues(
					valuesWithClassCounters);
			inOutCounts.moveOutToIn(missingCounters.get(0).classCounter);
			double insetScoreWithNull = scorer.scoreSplit(inOutCounts);
			return Pair.with(
					new CategoricalBranch(parent, attribute, Sets.newHashSet(MISSING_VALUE)),
					insetScoreWithNull);
		}

		Serializable lastValThatImprovedScore = null;

		int inSetSize = 0;
		int outSetSize = valuesWithClassCounters.size();
		for (final AttrValClassCounter attrValClassCounter : valuesWithClassCounters) {
			final Serializable value = attrValClassCounter.attrValue;
			final ClassCounter valClassCounts = attrValClassCounter.classCounter;
			inSetSize++;
			outSetSize--;

			// skip testing the split of no outset, as it is the same as doing no split, and
			// it will be handled separately
			if (outSetSize == 0) {
				continue;
			}

			// skip attribute value if no counts available
			// TODO: Not sure this works properly, create a good test
			if (valClassCounts == null) {
				continue;
			}

			// skips a value of a categorical variable if #occurences is below threshold
			// TODO: Not sure this works properly: create a good test
			if (isBelowMinAttributeOccurancesThreshold(valClassCounts)) {
				continue;
			}

			// handle ignored values: they need to be flipped from the outset to inset when
			// outset becomes smaller, see IgnoredValuesHandler javadoc for details.
			if (!ignoredValuesHandler.skip && !ignoredValuesHandler.isFlipped()
					&& outSetSize < inSetSize) {
				inOutCounts.moveOutToIn(ignoredValuesHandler.getCounter());
				ignoredValuesHandler.nowFlipped();
			}

			inOutCounts.moveOutToIn(valClassCounts);

			if (inOutCounts.totalsAreOverThreshold(minLeafInstances)
					&& setsSizesFitInMaxAllowed(inSetSize, outSetSize)
					&& scoreTracker.isBetter(inOutCounts)) {
				scoreTracker.setBest(inOutCounts);
				lastValThatImprovedScore = value;
			}
		}

		// check if doing no split is still better than the best found split
		if (scoreTracker.noSplitIsBetter()) {
			return null;
		}

		final Set<Serializable> returnSet = buildReturnSet(valuesWithClassCounters,
				lastValThatImprovedScore);
		if (returnSet.isEmpty()) {
			return null;
		}

		return Pair.with(new CategoricalBranch(parent, attribute, returnSet),
				scoreTracker.getBest());
	}

	private Set<Serializable> buildReturnSet(List<AttrValClassCounter> valuesWithClassCounters,
			Serializable lastValThatImprovedScore) {
		final Set<Serializable> inSet = Sets.newHashSet();
		final Set<Serializable> outSet = Sets.newHashSet();
		boolean insetIsBuiltNowBuildingOutset = false;
		for (AttrValClassCounter attrValClassCounter : valuesWithClassCounters) {
			if (!insetIsBuiltNowBuildingOutset) {
				inSet.add(attrValClassCounter.attrValue);
				if (attrValClassCounter.attrValue.equals(lastValThatImprovedScore)) {
					insetIsBuiltNowBuildingOutset = true;
				}
			} else {
				outSet.add(attrValClassCounter.attrValue);
			}
		}
		return outSet.size() < inSet.size() ? outSet : inSet;
	}

	private boolean setsSizesFitInMaxAllowed(int inSetSize, int outSetSize) {
		return inSetSize <= maxCategoricalInSetSize || outSetSize <= maxCategoricalInSetSize;
	}

	/**
	 * Returns a pair: global counter (target class -> #instances), and a sorted
	 * list of category values and associated target counters.
	 * 
	 */
	private Pair<ClassCounter, List<AttrValClassCounter>> getSortedValueClassCounters(
			String attribute, Iterable<? extends AbstractInstance> instances) {

		Pair<ClassCounter, List<AttrValClassCounter>> valueOutcomeCountsPairs = ClassCounter
				.getSortedListOfAttrValuesWithClassCounters(instances, attribute, splitAttribute,
						id, minorityClassification);

		return valueOutcomeCountsPairs;
	}

	private InOutCounts recalculateCountForInset(Set<Serializable> inSet, ClassCounter outCount,
			List<AttrValClassCounter> allValuesWithCount) {
		InOutCounts inOutCounts = new InOutCounts(new ClassCounter(), outCount);
		for (AttrValClassCounter valueWithCount : allValuesWithCount) {
			if (inSet.contains(valueWithCount.attrValue)) {
				inOutCounts.moveOutToIn(valueWithCount.classCounter);
			}
		}
		return inOutCounts;
	}

	private Pair<? extends Branch, Double> createNClassCategoricalNode(Branch parent,
			final String attribute, final Iterable<? extends AbstractInstance> instances) {

		final Set<Serializable> values = getAttributeValues(instances, attribute);

		if (insufficientTrainingDataGivenNumberOfAttributeValues(instances, values)) {
			return null;
		}

		final Set<Serializable> inValueSet = Sets.newHashSet(); // the in-set

		// the histogram of counts by classification for the in-set
		ClassCounter inSetClassificationCounts = new ClassCounter();

		final Pair<ClassCounter, Map<Serializable, ClassCounter>> valueOutcomeCountsPair = ClassCounter
				.countAllByAttributeValues(instances, attribute, splitAttribute, id);
		// global classification counter (all attribute values)
		ClassCounter outSetClassificationCounts = valueOutcomeCountsPair.getValue0();

		// map of attribute value _> classCounter
		final Map<Serializable, ClassCounter> valueOutcomeCounts = valueOutcomeCountsPair
				.getValue1();
		double insetScore = 0;

		// force split on missing values in case it is set by the TreeBuilder
		ClassCounter missingValuesCount = valueOutcomeCounts.get(MISSING_VALUE);
		if (forceSplitsOnMissing && missingValuesCount != null) {
			inValueSet.add(MISSING_VALUE);
			final ClassCounter testInCounts = inSetClassificationCounts.add(missingValuesCount);
			final ClassCounter testOutCounts = outSetClassificationCounts
					.subtract(missingValuesCount);
			double insetScoreWithNull = scorer.scoreSplit(testInCounts, testOutCounts);
			return Pair.with(new CategoricalBranch(parent, attribute, inValueSet),
					insetScoreWithNull);
		}

		while (true) {

			Optional<ScoreValuePair> bestValueAndScore = getBestScoreValuePair(
					inSetClassificationCounts, values, outSetClassificationCounts,
					valueOutcomeCounts);

			if (inValueSet.size() < maxCategoricalInSetSize && bestValueAndScore.isPresent()
					&& bestValueAndScore.get().getScore() > insetScore) {
				insetScore = bestValueAndScore.get().getScore();
				final Serializable bestValue = bestValueAndScore.get().getValue();
				inValueSet.add(bestValue);
				values.remove(bestValue);
				final ClassCounter bestValOutcomeCounts = valueOutcomeCounts.get(bestValue);
				inSetClassificationCounts = inSetClassificationCounts.add(bestValOutcomeCounts);
				outSetClassificationCounts = outSetClassificationCounts
						.subtract(bestValOutcomeCounts);

			} else {
				break;
			}
		}

		if (inSetClassificationCounts.getTotal() < minLeafInstances
				|| outSetClassificationCounts.getTotal() < minLeafInstances) {
			return null;
		}

		return Pair.with(new CategoricalBranch(parent, attribute, inValueSet), insetScore);
	}

	/**
	 * Given a set of attribute {@code values} (not sorted), iterate over them, on
	 * each iteration adding current value to the new set and calculating the score
	 * (based on the target counts) with this value in the set. Stop iterating when
	 * added value did not improve the score. Return the last value of the attribute
	 * which improved the score, and the total resulting score.
	 * 
	 * @param inSetClassificationCounts
	 * @param values
	 * @param outSetClassificationCounts
	 * @param valueOutcomeCounts
	 * @return
	 */
	private Optional<ScoreValuePair> getBestScoreValuePair(ClassCounter inSetClassificationCounts,
			Set<Serializable> values, ClassCounter outSetClassificationCounts,
			Map<Serializable, ClassCounter> valueOutcomeCounts) {

		Optional<ScoreValuePair> bestValueAndScore = Optional.empty();
		// values should be greater than 1
		for (final Serializable thisValue : values) {
			final ClassCounter testValCounts = valueOutcomeCounts.get(thisValue);

			// seems that original developer introduced this as a hotfix for negative
			// weights problem, but this generally should not happen
			if (testValCounts == null || thisValue == null) {
				continue;
			}

			if (isBelowMinAttributeOccurancesThreshold(testValCounts)) {
				continue;
			}
			final ClassCounter testInCounts = inSetClassificationCounts.add(testValCounts);
			final ClassCounter testOutCounts = outSetClassificationCounts.subtract(testValCounts);

			double scoreWithThisValueAddedToInset = scorer.scoreSplit(testInCounts, testOutCounts);

			if (!bestValueAndScore.isPresent()
					|| scoreWithThisValueAddedToInset > bestValueAndScore.get().getScore()) {
				bestValueAndScore = Optional
						.of(new ScoreValuePair(scoreWithThisValueAddedToInset, thisValue));
			}
		}
		return bestValueAndScore;
	}

	private boolean insufficientTrainingDataGivenNumberOfAttributeValues(
			final Iterable<? extends AbstractInstance> trainingData,
			final Set<Serializable> values) {
		final int averageInstancesPerValue = Iterables.size(trainingData) / values.size();
		return averageInstancesPerValue < this.minInstancesPerCategoricalVariable;
	}

	private Set<Serializable> getAttributeValues(
			final Iterable<? extends AbstractInstance> trainingData, final String attribute) {
		final Set<Serializable> values = Sets.newHashSet();
		for (final AbstractInstance instance : trainingData) {
			Serializable value = instance.getAttributes().get(attribute);
			if (value == null) {
				value = MISSING_VALUE;
			}
			values.add(value);
		}
		return values;
	}

	private boolean isBelowMinAttributeOccurancesThreshold(final ClassCounter testValCounts) {
		double totalCounts = testValCounts.getTotal();
		return totalCounts < minInstancesPerCategoricalVariable;
	}

	private Pair<? extends Branch, Double> createNumericNode(Branch parent, final String attribute,
			final Iterable<? extends AbstractInstance> instances, final double[] splits) {
		double bestScore = 0;
		double bestThreshold = 0;

		double lastThreshold = Double.MIN_VALUE;
		for (final double threshold : splits) {
			// Sometimes we can get a few thresholds the same, avoid wasted
			// effort when we do
			if (threshold == lastThreshold) {
				continue;
			}
			lastThreshold = threshold;

			final Iterable<? extends AbstractInstance> inSet = Iterables.filter(instances,
					new GreaterThanThresholdPredicate(attribute, threshold));
			final Iterable<? extends AbstractInstance> outSet = Iterables.filter(instances,
					new LessThanEqualThresholdPredicate(attribute, threshold));
			final ClassCounter inClassificationCounts = ClassCounter.countAll(inSet);
			final ClassCounter outClassificationCounts = ClassCounter.countAll(outSet);
			if (inClassificationCounts.getTotal() < minLeafInstances
					|| outClassificationCounts.getTotal() < minLeafInstances) {
				continue;
			}

			final double thisScore = scorer.scoreSplit(inClassificationCounts,
					outClassificationCounts);

			if (thisScore > bestScore) {
				bestScore = thisScore;
				bestThreshold = threshold;
			}
		}
		if (bestScore == 0) {
			return null;
		}
		return Pair.with(new NumericBranch(parent, attribute, bestThreshold), bestScore);
	}

	/**
	 * Iterate through tree until we get to a leaf. Using the training data indexes
	 * in the leaf and the training data provided build a tree from the leaf if
	 * possible. If a branch has only leaves as direct children, this will combine
	 * the data from the leaves and recreate the branch
	 *
	 * @param node The node we are attempting to further split
	 */
	private void splitNode(Node node, List<? extends AbstractInstance> trainingData) {
		if (node instanceof UpdatableLeaf) {
			UpdatableLeaf leaf = (UpdatableLeaf) node;
			if (leaf.parent != null) {
				Branch branch = (Branch) leaf.parent;
				Branch parent;
				Node toReplace;
				// determine if we are combining leaves and will be replacing the parent branch
				// or if we are replacing just this leaf
				if (shouldCombineData(branch)) {
					parent = (Branch) branch.parent;
					toReplace = branch;
				} else {
					parent = branch;
					toReplace = leaf;
				}
				Collection<AbstractInstance> leafData = getData(toReplace, trainingData);
				Node newNode = buildTree(parent, leafData, leaf.depth,
						createNumericSplits(leafData), node.isTrueChild());
				// replace the child that has the same reference as toReplace, intentionally
				// checking reference using ==
				if (parent.trueChild == toReplace) {
					parent.trueChild = newNode;
				} else {
					parent.falseChild = newNode;
				}
			}
		} else if (node instanceof Branch) {
			Branch branch = (Branch) node;
			splitNode(branch.trueChild, trainingData);
			// only split false child if we aren't combining leaves
			if (!shouldCombineData(branch)) {
				splitNode(branch.falseChild, trainingData);
			}

		}
	}

	private boolean shouldCombineData(Branch branch) {
		return branch.trueChild instanceof UpdatableLeaf
				&& branch.falseChild instanceof UpdatableLeaf && branch.parent != null;
	}

	/**
	 * @param node         a branch with UpdatableLeaf children or an UpdatableLeaf
	 * @param trainingData full set of trainingData
	 */
	private Collection<AbstractInstance> getData(Node node,
			List<? extends AbstractInstance> trainingData) {
		List<AbstractInstance> data = Lists.newArrayList();
		Collection<Integer> indexes = getIndexes(node);

		for (Integer index : indexes) {
			data.add(trainingData.get(index));
		}
		return data;
	}

	private Collection<Integer> getIndexes(Node node) {
		Collection<Integer> indexes = Collections.EMPTY_LIST;
		if (node instanceof UpdatableLeaf) {
			indexes = (((UpdatableLeaf) node).trainingDataIndexes);
		} else if (node instanceof Branch) {
			Branch branch = (Branch) node;
			indexes = ((UpdatableLeaf) branch.trueChild).trainingDataIndexes;
			indexes.addAll(((UpdatableLeaf) branch.falseChild).trainingDataIndexes);
		}
		return indexes;
	}

	private void addInstanceToNode(Node node, AbstractInstance instance) {
		if (node instanceof UpdatableLeaf) {
			UpdatableLeaf leaf = (UpdatableLeaf) node;
			leaf.addInstance(instance);
		} else if (node instanceof Branch) {
			Branch branch = (Branch) node;
			if (branch.getInPredicate().apply(instance)) {
				addInstanceToNode(branch.trueChild, instance);
			} else {
				addInstanceToNode(branch.falseChild, instance);
			}
		}
	}

	private void stripNode(Node node) {
		if (node instanceof UpdatableLeaf) {
			UpdatableLeaf leaf = (UpdatableLeaf) node;
			Branch branch = (Branch) leaf.parent;
			Leaf newLeaf = new Leaf(leaf.parent, leaf.classificationCounts, leaf.depth);
			if (branch.trueChild == node) {
				branch.trueChild = newLeaf;
			} else {
				branch.falseChild = newLeaf;
			}
		} else if (node instanceof Branch) {
			Branch branch = (Branch) node;
			stripNode(branch.trueChild);
			stripNode(branch.falseChild);
		}
	}

	public static class AttributeCharacteristics {
		public boolean isNumber = true;
	}

	private class GreaterThanThresholdPredicate implements Predicate<AbstractInstance> {

		private final String attribute;
		private final double threshold;

		public GreaterThanThresholdPredicate(String attribute, double threshold) {
			this.attribute = attribute;
			this.threshold = threshold;
		}

		@Override
		public boolean apply(@Nullable AbstractInstance input) {
			try {
				if (input == null) {// consider deleting
					return false;
				}
				Serializable value = input.getAttributes().get(attribute);
				if (value == null) {
					value = 0;
				}
				return ((Number) value).doubleValue() > threshold;
			} catch (final ClassCastException e) { // Kludge, need to
				// handle better
				return false;
			}
		}
	}

	private class LessThanEqualThresholdPredicate implements Predicate<AbstractInstance> {

		private final String attribute;
		private final double threshold;

		public LessThanEqualThresholdPredicate(String attribute, double threshold) {
			this.attribute = attribute;
			this.threshold = threshold;
		}

		@Override
		public boolean apply(@Nullable AbstractInstance input) {
			try {
				if (input == null) {
					return false;
				}
				Serializable value = input.getAttributes().get(attribute);
				if (value == null) {
					value = Double.MIN_VALUE;
				}
				// missing values should go the way of the outset. Future improvement shoud
				// allow missing values to go way of either inset or outset
				return ((Number) value).doubleValue() <= threshold;

				// Kludge, need to handle better
			} catch (final ClassCastException e) {
				return false;
			}
		}
	}

	private class ScoreValuePair {
		private double       score;
		private Serializable value;

		private ScoreValuePair(final double score, final Serializable value) {
			this.score = score;
			this.value = value;
		}

		public double getScore() {
			return score;
		}

		public Serializable getValue() {
			return value;
		}
	}

}
