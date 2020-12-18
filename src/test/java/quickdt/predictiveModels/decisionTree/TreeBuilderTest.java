package quickdt.predictiveModels.decisionTree;

import static com.google.common.math.DoubleMath.mean;
import static junit.framework.Assert.assertEquals;
import static junit.framework.Assert.assertFalse;
import static junit.framework.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;
import org.testng.internal.annotations.Sets;

import quickdt.Benchmarks;
import quickdt.Misc;
import quickdt.data.Attributes;
import quickdt.data.HashMapAttributes;
import quickdt.data.Instance;
import quickdt.predictiveModels.PredictiveModelWithDataBuilder;
import quickdt.predictiveModels.TreeBuilderTestUtils;
import quickdt.predictiveModels.decisionTree.scorers.SplitDiffScorer;
import quickdt.predictiveModels.decisionTree.tree.Branch;
import quickdt.predictiveModels.decisionTree.tree.CategoricalBranch;
import quickdt.predictiveModels.decisionTree.tree.Leaf;
import quickdt.predictiveModels.decisionTree.tree.Node;
import quickdt.predictiveModels.decisionTree.tree.NumericBranch;
import quickdt.predictiveModels.decisionTree.tree.Tree;

public class TreeBuilderTest {
	protected final Logger logger = LoggerFactory.getLogger(getClass());

	public final static String DELIMITER = Benchmarks.DELIMITER;

	@Test
	public void simpleBmiTest() throws Exception {
		final List<Instance> instances = TreeBuilderTestUtils.getInstances(10000);
		final TreeBuilder tb = new TreeBuilder(new SplitDiffScorer());
		final long startTime = System.currentTimeMillis();
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		TreeBuilderTestUtils.serializeDeserialize(node);

		final int nodeSize = node.size();
		Assert.assertTrue(nodeSize < 400, "Tree size should be less than 400 nodes");
		Assert.assertTrue((System.currentTimeMillis() - startTime) < 20000,
				"Building this node should take far less than 20 seconds");
	}

	@Test(enabled = false)
	public void multiScorerBmiTest() {
		final Set<Instance> instances = Sets.newHashSet();

		for (int x = 0; x < 10000; x++) {
			final double height = (4 * 12) + Misc.random.nextInt(3 * 12);
			final double weight = 120 + Misc.random.nextInt(110);
			final Instance instance = Instance.create(
					TreeBuilderTestUtils.bmiHealthy(weight, height), "weight", weight, "height",
					height);
			instances.add(instance);
		}
		{
			final TreeBuilder tb = new TreeBuilder(new SplitDiffScorer());
			final Tree tree = tb.buildPredictiveModel(instances);
			System.out.println("SplitDiffScorer node size: " + tree.node.size());
		}
	}

	@Test
	public void simpleBmiTestSplit() throws Exception {
		final List<Instance> instances = TreeBuilderTestUtils.getInstances(10000);
		final PredictiveModelWithDataBuilder<Tree> wb = getWrappedUpdatablePredictiveModelBuilder();
		wb.splitNodeThreshold(1);
		final long startTime = System.currentTimeMillis();
		final Tree tree = wb.buildPredictiveModel(instances);

		TreeBuilderTestUtils.serializeDeserialize(tree.node);

		int nodeSize = tree.node.size();
		double nodeMeanDepth = tree.node.meanDepth();
		Assert.assertTrue(nodeSize < 400, "Tree size should be less than 400 nodes");
		Assert.assertTrue(nodeMeanDepth < 6, "Mean depth should be less than 6");
		Assert.assertTrue((System.currentTimeMillis() - startTime) < 20000,
				"Building this node should take far less than 20 seconds");

		final List<Instance> newInstances = TreeBuilderTestUtils.getInstances(1000);
		final Tree newTree = wb.buildPredictiveModel(newInstances);
		Assert.assertTrue(tree == newTree, "Expect same tree to be updated");
		Assert.assertNotEquals(nodeSize, newTree.node.size(), "Expected new nodes");
		Assert.assertNotEquals(nodeMeanDepth, newTree.node.meanDepth(), "Expected new mean depth");

		nodeSize = newTree.node.size();
		nodeMeanDepth = newTree.node.meanDepth();
		wb.stripData(newTree);
		Assert.assertEquals(nodeSize, newTree.node.size(), "Expected same nodes");
		Assert.assertEquals(nodeMeanDepth, newTree.node.meanDepth(), "Expected same mean depth");
	}

	@Test
	public void simpleBmiTestNoSplit() throws Exception {
		final List<Instance> instances = TreeBuilderTestUtils.getInstances(10000);
		final PredictiveModelWithDataBuilder<Tree> wb = getWrappedUpdatablePredictiveModelBuilder();
		final long startTime = System.currentTimeMillis();
		final Tree tree = wb.buildPredictiveModel(instances);

		TreeBuilderTestUtils.serializeDeserialize(tree.node);

		int nodeSize = tree.node.size();
		double nodeMeanDepth = tree.node.meanDepth();
		Assert.assertTrue(nodeSize < 400, "Tree size should be less than 400 nodes");
		Assert.assertTrue(nodeMeanDepth < 6, "Mean depth should be less than 6");
		Assert.assertTrue((System.currentTimeMillis() - startTime) < 20000,
				"Building this node should take far less than 20 seconds");

		final List<Instance> newInstances = TreeBuilderTestUtils.getInstances(10000);
		final Tree newTree = wb.buildPredictiveModel(newInstances);
		Assert.assertTrue(tree == newTree, "Expect same tree to be updated");
		Assert.assertEquals(nodeSize, newTree.node.size(), "Expected same nodes");
		Assert.assertNotEquals(nodeMeanDepth, newTree.node.meanDepth(), "Expected new mean depth");

		nodeSize = newTree.node.size();
		nodeMeanDepth = newTree.node.meanDepth();
		wb.stripData(newTree);
		Assert.assertEquals(nodeSize, newTree.node.size(), "Expected same nodes");
		Assert.assertEquals(nodeMeanDepth, newTree.node.meanDepth(), "Expected same mean depth");
	}

	@Test
	public void simpleBmiTestRebuild() throws Exception {
		final List<Instance> instances = TreeBuilderTestUtils.getInstances(10000);
		final PredictiveModelWithDataBuilder<Tree> wb = getWrappedUpdatablePredictiveModelBuilder();
		wb.rebuildThreshold(1);
		final long startTime = System.currentTimeMillis();
		final Tree tree = wb.buildPredictiveModel(instances);

		TreeBuilderTestUtils.serializeDeserialize(tree.node);

		int nodeSize = tree.node.size();
		double nodeMeanDepth = tree.node.meanDepth();
		Assert.assertTrue(nodeSize < 400, "Tree size should be less than 400 nodes");
		Assert.assertTrue(nodeMeanDepth < 6, "Mean depth should be less than 6");
		Assert.assertTrue((System.currentTimeMillis() - startTime) < 20000,
				"Building this node should take far less than 20 seconds");

		final List<Instance> newInstances = TreeBuilderTestUtils.getInstances(10000);
		final Tree newTree = wb.buildPredictiveModel(newInstances);
		Assert.assertFalse(tree == newTree, "Expect new tree to be built");
	}

	private PredictiveModelWithDataBuilder<Tree> getWrappedUpdatablePredictiveModelBuilder() {
		final TreeBuilder tb = new TreeBuilder(new SplitDiffScorer());
		return new PredictiveModelWithDataBuilder<>(tb);
	}

	private void assertCategoricalBranchLimit(Branch root, int n) {
		if (root instanceof CategoricalBranch) {
			CategoricalBranch cb = (CategoricalBranch) root;
			Assert.assertTrue(cb.inSet.size() < n,
					"Split " + root + " has size of " + cb.inSet.size());
			logger.debug(cb.toString());
		}
		if (root.trueChild instanceof Branch) {
			assertCategoricalBranchLimit((Branch) root.trueChild, n);
		}
		if (root.falseChild instanceof Branch) {
			assertCategoricalBranchLimit((Branch) root.falseChild, n);
		}
	}

	private void logBranchRecursively(Branch branch) {
		logger.debug(branch.toString());
		logger.debug("true: {}; false: {}", branch.trueChild.toString(),
				branch.falseChild.toString());
		if (branch.trueChild instanceof Branch) {
			logBranchRecursively((Branch) branch.trueChild);
		}
		if (branch.falseChild instanceof Branch) {
			logBranchRecursively((Branch) branch.falseChild);
		}
	}

	public static List<Instance> loadCsvDataset(int classificationField, String fileName) {
		return loadCsvDataset(classificationField, fileName, new ArrayList<>());
	}

	public static List<Instance> loadCsvDataset(int classificationField, String fileName,
			List<Integer> numericColumns) {
		List<Instance> result = new LinkedList<>();
		String classification;
		Attributes attributes;
		String line;
		String[] header = null;
		assertFalse(numericColumns.contains(classificationField));

		try (BufferedReader br = new BufferedReader(new InputStreamReader(new GZIPInputStream(
				Benchmarks.class.getClassLoader().getResourceAsStream(fileName))))) {
			while ((line = br.readLine()) != null) {
				if (header == null) {
					header = line.split(DELIMITER);
					Assert.assertTrue(classificationField < header.length);
					continue;
				}
				String[] row = line.split(DELIMITER);
				Assert.assertEquals(header.length, row.length);
				attributes = new HashMapAttributes();
				classification = "";
				for (int idx = 0; idx < header.length; idx++) {
					if (idx == classificationField) {
						classification = row[idx];
					} else if (numericColumns.contains(idx)) {
						if (row[idx].equals("")) {
							attributes.put(header[idx], null);
						} else {
							attributes.put(header[idx], Double.valueOf(row[idx]));
						}
					} else {
						attributes.put(header[idx], row[idx]);
					}
				}
				if (!StringUtils.isEmpty(classification)) {
					result.add(new Instance(attributes, classification));
				}
			}
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
		return result;
	}

	// @Test
	// public void rpaTest() throws Exception {
	// final List<Instance> instances = loadRpaDataset(1,
	// "quickdt/synthetic/basic_mixed.csv.gz");
	// for (int n = 1; n < 10; n++) {
	// logger.debug("");
	// logger.debug("== Testing for {} categories ==", n);
	// // final TreeBuilder tb = new TreeBuilder(new SplitDiffScorer())
	// final TreeBuilder tb = new TreeBuilder()
	// // final TreeBuilder tb = new TreeBuilder(new GiniImpurityScorer())
	// .minimumScore(1e-12).maxCategoricalInSetSize(n);
	// final Tree tree = tb.buildPredictiveModel(instances);
	// final Node node = tree.node;
	//
	// TreeBuilderTestUtils.serializeDeserialize(node);
	//
	// assertCategoricalBranchLimit((Branch) node, n + 1);
	//
	// }
	// // Assert.assertEquals(11, node.size());
	// }

	private List<Instance> setNumericNullsToMean(List<Instance> instances,
			String[] attributeNames) {
		for (String attributeName : attributeNames) {
			List<Integer> nullRows = new ArrayList<>();
			List<Double> numericValues = new ArrayList<>();
			for (int rowCounter = 0; rowCounter < instances.size(); rowCounter++) {
				Serializable val = instances.get(rowCounter).getAttributes().get(attributeName);
				if (val == null) {
					nullRows.add(rowCounter);
				} else if (val instanceof Number) {
					numericValues.add((Double) val);
				} else {
					throw new RuntimeException(
							"Numeric column should not contain values other than Number and null");
				}
			}
			double mean = mean(numericValues);

			for (int nullRow : nullRows) {
				instances.get(nullRow).getAttributes().put(attributeName, mean);
			}
		}
		return instances;
	}

	@Test
	public void testBasicCategoricalSplit() throws IOException, ClassNotFoundException {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicCategorical.csv.gz", new ArrayList<>());
		for (int n = 1; n < 10; n++) {
			logger.debug("");
			logger.debug("== Testing for {} categories ==", n);
			// final TreeBuilder tb = new TreeBuilder(new SplitDiffScorer())
			final TreeBuilder tb = new TreeBuilder()
					// final TreeBuilder tb = new TreeBuilder(new GiniImpurityScorer())
					.minimumScore(1e-12).maxCategoricalInSetSize(n);
			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;

			TreeBuilderTestUtils.serializeDeserialize(node);

			assertCategoricalBranchLimit((Branch) node, n + 1);
		}
	}

	@Test
	public void testBasicNumericSingleSplit() {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicSingleSplitNumeric.csv.gz", Arrays.asList(0));
		for (int n = 1; n < 10; n++) {
			logger.debug("");
			logger.debug("== Testing for {} categories ==", n);
			// final TreeBuilder tb = new TreeBuilder(new SplitDiffScorer())
			final TreeBuilder tb = new TreeBuilder()
					// final TreeBuilder tb = new TreeBuilder(new GiniImpurityScorer())
					.minimumScore(1e-12).maxCategoricalInSetSize(n);
			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;

			logBranchRecursively((Branch) node);
		}
	}

	@Test
	public void testBasicNumericMultipleSplit() {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicMultipleSplitNumeric.csv.gz", Arrays.asList(0));
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12);
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		logBranchRecursively((Branch) node);
	}

	@Test
	public void testBasicNumericWithMissing() {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicNumericWithMissing.csv.gz", Arrays.asList(0));
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12);
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		logBranchRecursively((Branch) node);
	}

	@Test
	public void testBasicNumericWithMissingSetToMean() {

		List<Integer> numericColumns = Arrays.asList(0);
		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicNumericWithMissing.csv.gz", numericColumns);
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12);
		instances = setNumericNullsToMean(instances, new String[] { "NUM" });
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		logBranchRecursively((Branch) node);

		assertTrue(node instanceof NumericBranch);
		NumericBranch branch = (NumericBranch) node;
		assertEquals(2.5, branch.threshold);
	}

	@Test
	public void testBasicNumericWithMissingLimitCategory() {

		List<Integer> numericColumns = Arrays.asList(0);
		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicLargerNumericWithMissing.csv.gz", numericColumns);
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1);

		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		logBranchRecursively((Branch) node);

		// test first split on MISSING_VALUE
		assertTrue(node instanceof CategoricalBranch);
		CategoricalBranch branch = (CategoricalBranch) node;
		assertEquals(1, branch.inSet.size());
		assertTrue(branch.inSet.contains(TreeBuilder.MISSING_VALUE));

		// test second split is numeric, threshold on 2.0
		assertTrue(branch.falseChild instanceof NumericBranch);
		assertEquals(2.0, ((NumericBranch) branch.falseChild).threshold);

	}

	@Test
	public void testBasicNumericWithMissingInMajority() {

		List<Integer> numericColumns = Arrays.asList(0);
		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicNumericWithMissingInMajority.csv.gz", numericColumns);
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1);

		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		logBranchRecursively((Branch) node);

		assertTrue(node instanceof CategoricalBranch);
		CategoricalBranch branch = (CategoricalBranch) node;
		assertEquals(1, branch.inSet.size());
		assertTrue(branch.inSet.contains(0.0));
	}

	@Test
	public void testBasicNumericWithMissingInMajorityForceMissingSplit() {

		List<Integer> numericColumns = Arrays.asList(0);
		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicNumericWithMissingInMajority.csv.gz", numericColumns);
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1)
				.forceSplitOnNull();

		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		logBranchRecursively((Branch) node);

		assertTrue(node instanceof CategoricalBranch);
		CategoricalBranch branch = (CategoricalBranch) node;
		assertEquals(1, branch.inSet.size());
		assertTrue(branch.inSet.contains(TreeBuilder.MISSING_VALUE));
	}

	@Test
	public void testDoubleWithNaN() {
		// NaN values are not treated specially (as opposed to null values), therefore
		// the numeric variable with NaN is still treated as numeric; NaNs are driven
		// into the false nodes
		List<Integer> numericColumns = Arrays.asList(0);
		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicNumericWithNaNInMajority.csv.gz", numericColumns);
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1)
				.forceSplitOnNull();

		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		logBranchRecursively((Branch) node);

		assertTrue(node instanceof NumericBranch);
		NumericBranch branch = (NumericBranch) node;
		assertEquals(2.0, branch.threshold);

	}

	@Test
	public void testBasicCategoricalWithMissingInMinority() {

		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicCategoricalWithMissingInMinority.csv.gz");
		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1);// .forceSplitOnNull(true);
			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;

			logBranchRecursively((Branch) node);

			assertTrue(node instanceof CategoricalBranch);
			CategoricalBranch branch = (CategoricalBranch) node;
			assertEquals(1, branch.inSet.size());
			assertTrue(branch.inSet.contains(""));
		}

		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1)
					// forceSplitOnNull should not impact the categorical, because missing values
					// are represented as "", not null
					.forceSplitOnNull();
			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;
			logBranchRecursively((Branch) node);

			assertTrue(node instanceof CategoricalBranch);
			CategoricalBranch branch = (CategoricalBranch) node;
			assertEquals(1, branch.inSet.size());
			assertTrue(branch.inSet.contains(""));
		}

	}

	@Test
	public void testBasicCategoricalWithMissingInMajority() {

		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicCategoricalWithMissingInMajority.csv.gz");
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1)
				// forceSplitOnNull should not impact the categorical, because missing values
				// are represented as "", not null
				.forceSplitOnNull();

		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		logBranchRecursively((Branch) node);

		assertTrue(node instanceof CategoricalBranch);
		CategoricalBranch branch = (CategoricalBranch) node;
		assertEquals(1, branch.inSet.size());
		assertTrue(branch.inSet.contains("A"));
	}

	@Test
	public void testMulticlassNumeric() {
		// an example of inefficient splitting strategy: solves the problem with 4 leafs
		// instead of 3
		List<Integer> numericColumns = Arrays.asList(0);
		List<Instance> instances = loadCsvDataset(1, "quickdt/synthetic/multiclassNumeric.csv.gz",
				numericColumns);

		Consumer<Node> assertions = (Node node) -> {
			assertTrue(node instanceof NumericBranch);
			NumericBranch branch = (NumericBranch) node;
			assertEquals(4.0, branch.threshold);

			assertTrue(branch.trueChild instanceof NumericBranch);
			assertEquals(7.0, ((NumericBranch) branch.trueChild).threshold);

			assertTrue(branch.falseChild instanceof NumericBranch);
			assertEquals(3.0, ((NumericBranch) branch.falseChild).threshold);
		};

		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1);
			// .forceSplitOnNull(true);

			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;

			logBranchRecursively((Branch) node);

			assertions.accept(node);
		}

		{ // test that splits on null have no effect
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1)
					// allow splits on null has no effect
					.forceSplitOnNull();

			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;

			// logBranchRecursively((Branch) node);

			assertions.accept(node);
		}
	}

	@Test
	public void testMulticlassNumericWithNull() {
		List<Integer> numericColumns = Arrays.asList(0);
		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/multiclassNumericWithMissing.csv.gz", numericColumns);

		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(3);

		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		logBranchRecursively((Branch) node);

		assertTrue(node instanceof CategoricalBranch);
		CategoricalBranch branch = (CategoricalBranch) node;
		assertTrue(branch.inSet.contains(5.0));
		assertTrue(branch.inSet.contains(6.0));
		assertEquals(3, branch.inSet.size());
	}

	@Test
	public void testMulticlassNumericWithNullInMinorityForceSplitOnNull() {
		List<Integer> numericColumns = Arrays.asList(0);
		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/multiclassNumericWithMissing.csv.gz", numericColumns);

		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1);

			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;

			// logBranchRecursively((Branch) node);

			assertEquals(1, ((CategoricalBranch) node).inSet.size());
			assertTrue(((CategoricalBranch) node).inSet.contains(TreeBuilder.MISSING_VALUE));
			assertEquals(4.0, ((NumericBranch) ((CategoricalBranch) node).falseChild).threshold);
		}

	}

	@Test
	public void testMulticlassNumericWithNullInMajorityForceSplitOnNull() {
		List<Integer> numericColumns = Arrays.asList(0);
		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/multiclassNumericWithMissingInMajority.csv.gz", numericColumns);

		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2);

			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;

			// logBranchRecursively((Branch) node);

			assertEquals(2, ((CategoricalBranch) node).inSet.size());
			assertTrue(((CategoricalBranch) node).inSet.contains(0.0));
		}

		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(3)
					.forceSplitOnNull();

			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;

			// logBranchRecursively((Branch) node);

			assertEquals(1, ((CategoricalBranch) node).inSet.size());
			assertTrue(((CategoricalBranch) node).inSet.contains(TreeBuilder.MISSING_VALUE));
			assertEquals(6.0, ((NumericBranch) ((CategoricalBranch) node).falseChild).threshold);
		}
	}

	private int getMaxDepth(Node node) {
		Leaf leaf = node.collectLeaves().stream().max((x, y) -> x.depth - y.depth).get();
		return leaf.depth;
	}

	@Test
	public void testReduceDepth() {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicCategorical.csv.gz");
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1);
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node root = tree.node;

		for (int depth = getMaxDepth(root); depth > 1; depth--) {
			Assert.assertEquals(depth, getMaxDepth(root));
			tree.collapseDeepestLeaves(false);
			logger.debug("");
			logBranchRecursively((Branch) root);
		}
	}

	@Test
	public void testShowLeafMetrics() {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicLargerNumericWithMissing.csv.gz");
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2);
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node root = tree.node;

		List<Leaf> leaves = root.collectLeaves();
		logger.debug(leaves.stream().map(Leaf::toString).collect(Collectors.joining("\n\n")));
	}

	@Test
	public void testMetricsBasicNumeric() {

		List<Integer> numericColumns = Arrays.asList(0);
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicLargerNumericWithMissing.csv.gz", numericColumns);
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2);

		final Tree tree = tb.buildPredictiveModel(instances);
		final Branch root = (Branch) tree.node;

		logBranchRecursively(root);

		Map<Serializable, Double> recall = tree.getRecall();
		// System.out.println(recall);
		assertTrue(recall.get("0") - 1 < 0.001);
		assertTrue(recall.get("1") - 1 < 0.001);

		Map<Serializable, Double> precision = tree.getPrecision();
		// System.out.println(precision);
		assertTrue(precision.get("0") - 1 < 0.001);
		assertTrue(precision.get("1") - 1 < 0.001);
	}

	@Test
	public void testMetricsMixedCategorical() {

		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/categoricalMixed.csv.gz");
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2);

		final Tree tree = tb.buildPredictiveModel(instances);
		final Branch root = (Branch) tree.node;

		// logBranchRecursively(root);

		Map<Serializable, Double> recall = tree.getRecall();
		assertTrue(recall.get("0") - 0.833 < 0.001);
		assertTrue(recall.get("1") - 0.833 < 0.001);

		Map<Serializable, Double> precision = tree.getPrecision();
		assertTrue(precision.get("0") - 0.833 < 0.001);
		assertTrue(precision.get("1") - 0.833 < 0.001);

	}

	@Test
	public void testMetricsMixedCategorical2() {

		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/categoricalMixed2.csv.gz");
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2);

		final Tree tree = tb.buildPredictiveModel(instances);
		final Branch root = (Branch) tree.node;

		// logBranchRecursively(root);

		Map<Serializable, Double> recall = tree.getRecall();
		assertTrue(recall.get("0") - 0.777 < 0.001);
		assertTrue(recall.get("1") - 0.833 < 0.001);

		Map<Serializable, Double> precision = tree.getPrecision();
		assertTrue(precision.get("0") - 0.875 < 0.001);
		assertTrue(precision.get("1") - 0.714 < 0.001);
	}

	@Test
	public void testMetricsMulticlassCategorical() {

		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/categoricalMulticlassMixed.csv.gz");
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2);

		final Tree tree = tb.buildPredictiveModel(instances);
		final Branch root = (Branch) tree.node;

		logBranchRecursively(root);

		// for (Leaf leaf : tree.getLeaves()) {
		// System.out.println(leaf);
		// System.out.println(leaf.getMajorityClass());
		// }

		Map<Serializable, Double> recall = tree.getRecall();
		// System.out.println(recall);
		assertTrue(recall.get("0") - 0.777 < 0.001);
		assertTrue(recall.get("1") - 0.833 < 0.001);
		assertTrue(recall.get("2") - 0.571 < 0.001);

		Map<Serializable, Double> precision = tree.getPrecision();
		// System.out.println(precision);
		assertTrue(precision.get("0") - 0.777 < 0.001);
		assertTrue(precision.get("1") - 0.555 < 0.001);
		assertTrue(precision.get("2") - 1 < 0.001);

		Map<Serializable, Map<Serializable, Double>> falsePositiveDistribution = tree
				.getFalsePositiveDistribution();
		assertEquals(tree.getClassCounter().allClassifications().size(),
				falsePositiveDistribution.size());
		for (Serializable next : tree.getClassCounter().allClassifications()) {
			assertFalse(falsePositiveDistribution.get(next).containsKey(next));
		}

		Map<Serializable, Map<Serializable, Double>> falseNegativeDistribution = tree
				.getFalseNegativeDistribution();
		assertEquals(tree.getClassCounter().allClassifications().size(),
				falseNegativeDistribution.size());
		for (Serializable next : tree.getClassCounter().allClassifications()) {
			assertFalse(falseNegativeDistribution.get(next).containsKey(next));
		}

		double totalFalseNegatives = falsePositiveDistribution.entrySet().stream()
				.mapToDouble(
						e -> e.getValue().values().stream().mapToDouble(Double::doubleValue).sum())
				.sum();
		double totalFalsePositives = falseNegativeDistribution.entrySet().stream()
				.mapToDouble(
						e -> e.getValue().values().stream().mapToDouble(Double::doubleValue).sum())
				.sum();
		assertEquals(totalFalseNegatives, totalFalsePositives);
	}

	@Test
	public void testMetricsMulticlassNumeric() {
		List<Integer> numericColumns = Arrays.asList(0);

		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/numericMulticlassMixed.csv.gz", numericColumns);
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2);

		Tree tree = tb.buildPredictiveModel(instances);

		// final Branch root = (Branch) tree.node;
		// logBranchRecursively(root);
		//
		// for (Leaf leaf : tree.getLeaves()) {
		// System.out.println(leaf);
		// System.out.println(leaf.getMajorityClass());
		// }

		Map<Serializable, Double> recall = tree.getRecall();
		System.out.println(recall);
		assertTrue(recall.get("0") - 1 < 0.001);
		assertTrue(recall.get("1") - 0.571 < 0.001);
		assertTrue(recall.get("2") - 0.9 < 0.001);

		Map<Serializable, Double> precision = tree.getPrecision();
		System.out.println(precision);
		assertTrue(precision.get("0") - 0.7 < 0.001);
		assertTrue(precision.get("1") - 0.8 < 0.001);
		assertTrue(precision.get("2") - 1 < 0.001);

		Map<Serializable, Map<Serializable, Double>> falsePositiveDistribution = tree
				.getFalsePositiveDistribution();
		assertEquals(tree.getClassCounter().allClassifications().size(),
				falsePositiveDistribution.size());
		for (Serializable next : tree.getClassCounter().allClassifications()) {
			assertFalse(falsePositiveDistribution.get(next).containsKey(next));
		}

		Map<Serializable, Map<Serializable, Double>> falseNegativeDistribution = tree
				.getFalseNegativeDistribution();
		assertEquals(tree.getClassCounter().allClassifications().size(),
				falseNegativeDistribution.size());
		for (Serializable next : tree.getClassCounter().allClassifications()) {
			assertFalse(falseNegativeDistribution.get(next).containsKey(next));
		}

		double totalFalseNegatives = falsePositiveDistribution.entrySet().stream()
				.mapToDouble(
						e -> e.getValue().values().stream().mapToDouble(Double::doubleValue).sum())
				.sum();
		double totalFalsePositives = falseNegativeDistribution.entrySet().stream()
				.mapToDouble(
						e -> e.getValue().values().stream().mapToDouble(Double::doubleValue).sum())
				.sum();
		assertEquals(totalFalseNegatives, totalFalsePositives);
	}

	@Test
	public void testCategoricalMulticlassBinaryPruning() {

		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/categoricalMulticlassMixed.csv.gz");
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2);

		Tree tree = tb.buildPredictiveModel(instances);

		logBranchRecursively((Branch) tree.node);

		assertEquals(8, tree.getLeaves().size());

		tree = tree.pruneSameCategoryLeaves();

		assertEquals(5, tree.getLeaves().size());

		// assertTrue(node instanceof CategoricalBranch);
		// CategoricalBranch branch = (CategoricalBranch) node;
		// assertEquals(1, branch.inSet.size());
		// assertTrue(branch.inSet.contains("A"));
	}

	@Test
	public void testCategoricalMulticlassBinaryPruningNoSplit() {

		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/categoricalMulticlassMixed.csv.gz");
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2)
				.maxDepth(0);
		Tree tree = tb.buildPredictiveModel(instances);
		assertEquals(1, tree.getLeaves().size());
		tree = tree.pruneSameCategoryLeaves();
		assertEquals(1, tree.getLeaves().size());
		tree.collapseDeepestLeaves(true);

		// logBranchRecursively((Branch) tree.node);

		// assertEquals(5, tree.getLeaves().size());

		// assertTrue(node instanceof CategoricalBranch);
		// CategoricalBranch branch = (CategoricalBranch) node;
		// assertEquals(1, branch.inSet.size());
		// assertTrue(branch.inSet.contains("A"));
	}

	@Test
	public void testCategoricalMulticlassBinaryPruningLowDepth() {

		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/categoricalMulticlassMixed.csv.gz");
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2)
				.maxDepth(1);

		Tree tree = tb.buildPredictiveModel(instances);

		assertEquals(2, tree.getLeaves().size());
		tree = tree.pruneSameCategoryLeaves();

		assertEquals(2, tree.getLeaves().size());
	}

	@Test
	public void testSameCategoryPruningBuilder() {

		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/categoricalMulticlassMixed.csv.gz");
		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2);
			Tree tree = tb.buildPredictiveModel(instances);
			assertEquals(8, tree.getLeaves().size());
		}
		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(2)
					.pruneSameCategory();
			Tree tree = tb.buildPredictiveModel(instances);
			assertEquals(5, tree.getLeaves().size());
		}
	}

	@Test
	public void testGateway() {
		List<Instance> instances = loadCsvDataset(7, "quickdt/gateway.csv.gz");
		final TreeBuilder tb = new TreeBuilder().forceSplitOnNull().maxCategoricalInSetSize(3)
				.maxDepth(5).minLeafInstances(10);
		Tree tree = tb.buildPredictiveModel(instances);
		assertTrue(tree.node instanceof CategoricalBranch);
		assertEquals(71.0, ((Branch) tree.node).trueChild.getClassificationCounter().getTotal());
	}

	@Test
	public void testUnclassifiableCategories() {

		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/categoricalSomeCategoriesCannotBeClassified.csv.gz");

		final TreeBuilder tb = new TreeBuilder().maxCategoricalInSetSize(2);
		Tree tree = tb.buildPredictiveModel(instances);

		assertEquals("{0=5.0, 1=5.0, 2=0.0}", tree.getTruePositiveCounts().toString());
		assertEquals("{0=0.9090909090909091, 1=0.9090909090909091, 2=0.0}",
				tree.getF1().toString());
	}

	/**
	 * Test triggered by an error in the external application, using quickdt
	 */
	@Test
	public void testNumericSplit1() {

		List<Integer> numericColumns = Arrays.asList(1);
		final List<Instance> instances = loadCsvDataset(0, "quickdt/synthetic/numericSplit1.csv.gz",
				numericColumns);

		// test with settings used in the external application
		int minLeafInstances = (int) Math.ceil(0.05 * instances.size());
		final TreeBuilder tb = new TreeBuilder().forceSplitOnNull().maxCategoricalInSetSize(3)
				.maxDepth(5).minInstancesPerCategoricalVariable(1)
				.minLeafInstances(minLeafInstances).pruneSameCategory();

		Tree tree = tb.buildPredictiveModel(instances);

		assertTrue(tree.node instanceof NumericBranch);
		assertEquals("var > 1.0", tree.node.toString());
	}

	/**
	 * Test triggered by an error in the external application, using quickdt
	 */
	@Test
	public void testNumericSplit2() {

		List<Integer> numericColumns = Arrays.asList(0, 1);
		final List<Instance> instances = loadCsvDataset(2, "quickdt/synthetic/numericSplit2.csv.gz",
				numericColumns);

		// test with settings used in the external application
		int minLeafInstances = (int) Math.ceil(0.05 * instances.size());
		final TreeBuilder tb = new TreeBuilder().forceSplitOnNull().maxCategoricalInSetSize(3)
				.maxDepth(5).minInstancesPerCategoricalVariable(1)
				.minLeafInstances(minLeafInstances).pruneSameCategory();

		Tree tree = tb.buildPredictiveModel(instances);
		assertTrue(tree.node instanceof NumericBranch);
		assertEquals("CAT2[B] > 1.0", tree.node.toString());
	}

	@Test
	public void testUniqueIDHandling() {
		final List<Instance> instances = loadCsvDataset(3, "quickdt/synthetic/unique_id.csv.gz");

		// test with settings used in the external application
		int minLeafInstances = (int) Math.ceil(0.05 * instances.size());
		final TreeBuilder tb = new TreeBuilder().forceSplitOnNull().maxCategoricalInSetSize(1)
				.maxDepth(1).minInstancesPerCategoricalVariable(1)
				.minLeafInstances(minLeafInstances).pruneSameCategory().setIdAttribute("ID");

		Tree tree = tb.buildPredictiveModel(instances);

		List<Leaf> leaves = tree.getLeaves();
		assertEquals("{0=3, 1=3}", tb.getIdAttributeHandler().getTotalCounts().toString());
		for (Leaf leaf : leaves) {
			assertEquals(Integer.valueOf(3),
					tb.getIdAttributeHandler().getCountForLeafClass(leaf, "0"));

			assertEquals(Integer.valueOf(3),
					tb.getIdAttributeHandler().getCountForMajorityClass(leaf));
		}
		assertEquals("CAT in [0]", tree.node.toString());
	}

	@Test
	public void testIgnoreMissing() {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicCategoricalWithMissingInMinority.csv.gz");
		{
			// test with settings used in the external application
			int minLeafInstances = (int) Math.ceil(0.05 * instances.size());
			final TreeBuilder tb = new TreeBuilder().forceSplitOnNull().maxCategoricalInSetSize(1)
					.maxDepth(1).minInstancesPerCategoricalVariable(1)
					.minLeafInstances(minLeafInstances).pruneSameCategory();

			Tree tree = tb.buildPredictiveModel(instances);

			assertEquals("CAT in []", tree.node.toString());
		}

		{
			// test with settings used in the external application
			int minLeafInstances = (int) Math.ceil(0.05 * instances.size());
			final TreeBuilder tb = new TreeBuilder().forceSplitOnNull().maxCategoricalInSetSize(1)
					.maxDepth(1).minInstancesPerCategoricalVariable(1)
					.minLeafInstances(minLeafInstances).pruneSameCategory().ignoreEmptyStrings();

			Tree tree = tb.buildPredictiveModel(instances);

			assertEquals("CAT in [E]", tree.node.toString());
		}
	}

	@Test
	public void testBasicCategoricalWithMissingInMajorityIgnore() {

		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicCategoricalWithMissingInMajority.csv.gz");

		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1)
					// ignoreEmptyString
					.forceSplitOnNull().ignoreEmptyStrings();

			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;
			assertTrue(node instanceof CategoricalBranch);
			CategoricalBranch branch = (CategoricalBranch) node;
			assertEquals(1, branch.inSet.size());
			assertTrue(branch.inSet.contains("A"));
		}
		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1)
					// try ignoreValue
					.forceSplitOnNull().ignoreValue("");

			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;
			assertTrue(node instanceof CategoricalBranch);
			CategoricalBranch branch = (CategoricalBranch) node;
			assertEquals(1, branch.inSet.size());
			assertTrue(branch.inSet.contains("A"));
		}
		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1)
					// try doubling the values
					.forceSplitOnNull().ignoreValue("").ignoreValue("");

			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;
			assertTrue(node instanceof CategoricalBranch);
			CategoricalBranch branch = (CategoricalBranch) node;
			assertEquals(1, branch.inSet.size());
			assertTrue(branch.inSet.contains("A"));
		}
		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1)
					// try doubling the values
					.forceSplitOnNull().ignoreValues(Arrays.asList("", ""));

			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;
			assertTrue(node instanceof CategoricalBranch);
			CategoricalBranch branch = (CategoricalBranch) node;
			assertEquals(1, branch.inSet.size());
			assertTrue(branch.inSet.contains("A"));
		}
		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1)
					// forbid the best value
					.forceSplitOnNull().ignoreValues(Arrays.asList("A"));

			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;
			assertTrue(node instanceof CategoricalBranch);
			CategoricalBranch branch = (CategoricalBranch) node;
			assertEquals(1, branch.inSet.size());
			assertTrue(branch.inSet.contains("B"));
			assertEquals("[CAT in [B]->\n" + "0=1.0 (matches=1.0; contaminations=0.0)\n"
					+ ", CAT in [C]->CAT not in [B]->\n"
					+ "0=1.0 (matches=1.0; contaminations=0.0)\n"
					+ ", CAT in [D]->CAT not in [C]->CAT not in [B]->\n"
					+ "0=1.0 (matches=1.0; contaminations=0.0)\n"
					+ ", CAT in []->CAT not in [D]->CAT not in [C]->CAT not in [B]->\n"
					+ "1=1.0 (matches=3.0; contaminations=0.0)\n"
					+ ", CAT in [E]->CAT not in []->CAT not in [D]->CAT not in [C]->CAT not in [B]->\n"
					+ "1=1.0 (matches=1.0; contaminations=0.0)\n"
					+ ", CAT in [F]->CAT not in [E]->CAT not in []->CAT not in [D]->CAT not in [C]->CAT not in [B]->\n"
					+ "1=1.0 (matches=1.0; contaminations=0.0)\n"
					+ ", CAT not in [F]->CAT not in [E]->CAT not in []->CAT not in [D]->CAT not in [C]->CAT not in [B]->\n"
					+ "0=0.5 (matches=1.0; contaminations=1.0)\n"
					+ "1=0.5 (matches=1.0; contaminations=1.0)\n" + "]",
					tree.getLeaves().toString());
		}
		{
			final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1)
					// forbid all good values
					.forceSplitOnNull().ignoreValues(Arrays.asList("A", "B", "C", "D"));

			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;
			assertTrue(node instanceof CategoricalBranch);
			CategoricalBranch branch = (CategoricalBranch) node;
			assertEquals(1, branch.inSet.size());
			assertTrue(branch.inSet.contains(""));

			assertEquals(
					"[CAT in []->\n" + "1=1.0 (matches=3.0; contaminations=0.0)\n"
							+ ", CAT in [E]->CAT not in []->\n"
							+ "1=1.0 (matches=1.0; contaminations=0.0)\n"
							+ ", CAT in [F]->CAT not in [E]->CAT not in []->\n"
							+ "1=1.0 (matches=1.0; contaminations=0.0)\n"
							+ ", CAT not in [F]->CAT not in [E]->CAT not in []->\n"
							+ "0=0.8 (matches=4.0; contaminations=1.0)\n"
							+ "1=0.2 (matches=4.0; contaminations=1.0)\n" + "]",
					tree.getLeaves().toString());
		}
	}

	@Test
	public void testTreatNumericAsCategorical() {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicMultipleSplitNumeric.csv.gz", Arrays.asList(0));
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).treatNumericAsCategorical();
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		assertEquals(
				"[NUM in [4.0, 5.0, 3.0]->\n" + "1=1.0 (matches=3.0; contaminations=0.0)\n"
						+ ", NUM not in [4.0, 5.0, 3.0]->\n"
						+ "0=1.0 (matches=6.0; contaminations=0.0)\n" + "]",
				tree.getLeaves().toString());
	}

	@Test
	public void testFindOptimalSplitOnNumeric() {
		int maxTreeDepth = 5;
		int maxCategoricalInSetSize = 3;
		int minPerVarInstances = 1;

		final TreeBuilder tb = new TreeBuilder().forceSplitOnNull()
				.maxCategoricalInSetSize(maxCategoricalInSetSize).maxDepth(maxTreeDepth)
				.minInstancesPerCategoricalVariable(minPerVarInstances).pruneSameCategory()
				.numericTestSplits(10);

		final List<Instance> instances = loadCsvDataset(0,
				"quickdt/synthetic/optimalSplitOnNumeric.csv.gz", Arrays.asList(1));

		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		assertEquals(
				"[Amount > 1000.0->\n" + "manual approve=1.0 (matches=9.0; contaminations=0.0)\n"
						+ ", Amount <= 1000.0->\n"
						+ "auto approve=1.0 (matches=6.0; contaminations=0.0)\n" + "]",
				tree.getLeaves().toString());
	}

	@Test
	public void testVeryManyNumericSplits() {
		int maxTreeDepth = 5;
		int maxCategoricalInSetSize = 3;
		int minPerVarInstances = 1;

		final TreeBuilder tb = new TreeBuilder().forceSplitOnNull()
				.maxCategoricalInSetSize(maxCategoricalInSetSize).maxDepth(maxTreeDepth)
				.minInstancesPerCategoricalVariable(minPerVarInstances).pruneSameCategory()
				.numericTestSplits(999);

		final List<Instance> instances = loadCsvDataset(0,
				"quickdt/synthetic/optimalSplitOnNumeric.csv.gz", Arrays.asList(1));

		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		assertEquals(
				"[Amount > 1000.0->\n" + "manual approve=1.0 (matches=9.0; contaminations=0.0)\n"
						+ ", Amount <= 1000.0->\n"
						+ "auto approve=1.0 (matches=6.0; contaminations=0.0)\n" + "]",
				tree.getLeaves().toString());
	}

	@Test
	public void testIncorrectAssignmentOfEquals() {
		// this tests suboptimal splits on a numeric variable, in case when
		// numericTestSplits is low; a failure of this test is expected if the algorithm
		// will be improved.
		int maxTreeDepth = 5;
		int maxCategoricalInSetSize = 3;
		int minPerVarInstances = 1;

		final TreeBuilder tb = new TreeBuilder().forceSplitOnNull()
				.maxCategoricalInSetSize(maxCategoricalInSetSize).maxDepth(maxTreeDepth)
				.minInstancesPerCategoricalVariable(minPerVarInstances).pruneSameCategory()
				.numericTestSplits(5);

		final List<Instance> instances = loadCsvDataset(0,
				"quickdt/synthetic/optimalSplitOnNumeric.csv.gz", Arrays.asList(1));

		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		assertEquals(
				"[Amount > 1100.0->\n" + "manual approve=1.0 (matches=8.0; contaminations=0.0)\n"
						+ ", Amount > 800.0->Amount <= 1100.0->\n"
						+ "manual approve=0.5 (matches=1.0; contaminations=1.0)\n"
						+ "auto approve=0.5 (matches=1.0; contaminations=1.0)\n"
						+ ", Amount <= 800.0->Amount <= 1100.0->\n"
						+ "auto approve=1.0 (matches=5.0; contaminations=0.0)\n" + "]",
				tree.getLeaves().toString());

	}
}
