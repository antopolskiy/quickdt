package quickdt.predictiveModels.decisionTree;

import static com.google.common.math.DoubleMath.mean;
import static junit.framework.Assert.assertFalse;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
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

//	@Test
//	public void rpaTest() throws Exception {
//		final List<Instance> instances = loadRpaDataset(1, "quickdt/synthetic/basic_mixed.csv.gz");
//		for (int n = 1; n < 10; n++) {
//			logger.debug("");
//			logger.debug("== Testing for {} categories ==", n);
//			// final TreeBuilder tb = new TreeBuilder(new SplitDiffScorer())
//			final TreeBuilder tb = new TreeBuilder()
//					// final TreeBuilder tb = new TreeBuilder(new GiniImpurityScorer())
//					.minimumScore(1e-12).maxCategoricalInSetSize(n).smallTrainingSetLimit(2);
//			final Tree tree = tb.buildPredictiveModel(instances);
//			final Node node = tree.node;
//
//			TreeBuilderTestUtils.serializeDeserialize(node);
//
//			assertCategoricalBranchLimit((Branch) node, n + 1);
//
//		}
//		// Assert.assertEquals(11, node.size());
//	}

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
					.minimumScore(1e-12).maxCategoricalInSetSize(n).smallTrainingSetLimit(2);
			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;

			TreeBuilderTestUtils.serializeDeserialize(node);

			assertCategoricalBranchLimit((Branch) node, n + 1);
		}
	}

	@Test
	public void testBasicNumericSingleSplit() throws IOException, ClassNotFoundException {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicSingleSplitNumeric.csv.gz", Arrays.asList(0));
		for (int n = 1; n < 10; n++) {
			logger.debug("");
			logger.debug("== Testing for {} categories ==", n);
			// final TreeBuilder tb = new TreeBuilder(new SplitDiffScorer())
			final TreeBuilder tb = new TreeBuilder()
					// final TreeBuilder tb = new TreeBuilder(new GiniImpurityScorer())
					.minimumScore(1e-12).maxCategoricalInSetSize(n).smallTrainingSetLimit(2);
			final Tree tree = tb.buildPredictiveModel(instances);
			final Node node = tree.node;

			TreeBuilderTestUtils.serializeDeserialize(node);

			logBranchRecursively((Branch) node);
		}
	}

	@Test
	public void testBasicNumericMultipleSplit() throws IOException, ClassNotFoundException {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicMultipleSplitNumeric.csv.gz", Arrays.asList(0));
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).smallTrainingSetLimit(2);
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		TreeBuilderTestUtils.serializeDeserialize(node);

		logBranchRecursively((Branch) node);
	}

	@Test
	public void testBasicNumericWithNaN() throws IOException, ClassNotFoundException {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicNumericWithNaN.csv.gz", Arrays.asList(0));
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).smallTrainingSetLimit(2);
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		TreeBuilderTestUtils.serializeDeserialize(node);

		logBranchRecursively((Branch) node);
	}

	@Test
	public void testBasicNumericWithNaNSetToMean() throws IOException, ClassNotFoundException {

		List<Integer> numericColumns = Arrays.asList(0);
		List<Instance> instances = loadCsvDataset(1, "quickdt/synthetic/basicNumericWithNaN.csv.gz",
				numericColumns);
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).smallTrainingSetLimit(2);
		instances = setNumericNullsToMean(instances, new String[] { "NUM" });
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		TreeBuilderTestUtils.serializeDeserialize(node);

		logBranchRecursively((Branch) node);
	}

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
	public void testBasicNumericWithNaNLimitCategory() throws IOException, ClassNotFoundException {

		List<Integer> numericColumns = Arrays.asList(0);
		List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicLargerNumericWithNaN.csv.gz", numericColumns);
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).smallTrainingSetLimit(2)
				.maxCategoricalInSetSize(2);

//		instances = setNumericNullsToMean(instances, new String[] { "NUM" });
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node node = tree.node;

		TreeBuilderTestUtils.serializeDeserialize(node);

		logBranchRecursively((Branch) node);
	}

	public int getMaxDepth(Node node) {
		Leaf leaf = node.collectLeaves().stream().max((x, y) -> x.depth - y.depth).get();
		return leaf.depth;
	}

	@Test
	public void testReduceDepth() throws Exception {
		final List<Instance> instances = loadCsvDataset(1,
				"quickdt/synthetic/basicCategorical.csv.gz");
		final TreeBuilder tb = new TreeBuilder().minimumScore(1e-12).maxCategoricalInSetSize(1);
		final Tree tree = tb.buildPredictiveModel(instances);
		final Node root = tree.node;

//		TreeBuilderTestUtils.serializeDeserialize(root);

		for (int depth = getMaxDepth(root); depth > 1; depth--) {
			Assert.assertEquals(depth, getMaxDepth(root));
			tree.pruneDeepestLeaves();
			logger.debug("");
			logBranchRecursively((Branch) root);

		}
	}

}
