package quickdt.predictiveModelOptimizer;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

import quickdt.Benchmarks;
import quickdt.crossValidation.LogCrossValLossFunction;
import quickdt.crossValidation.StationaryCrossValidator;
import quickdt.data.AbstractInstance;
import quickdt.predictiveModels.PredictiveModelWithDataBuilder;
import quickdt.predictiveModels.PredictiveModelWithDataBuilderBuilder;
import quickdt.predictiveModels.randomForest.RandomForestBuilder;
import quickdt.predictiveModels.randomForest.RandomForestBuilderBuilder;

/**
 * Created by ian on 3/1/14.
 */
public class PredictiveModelOptimizerTest {
	private static final Logger logger = LoggerFactory
			.getLogger(PredictiveModelOptimizerTest.class);

	@Test
	public void irisTest() throws IOException {
		final List<AbstractInstance> instances = Benchmarks.loadIrisDataset();
		testWithTrainingSet(instances);
	}

	@Test(enabled = false)
	public void diabetesTest() throws IOException {
		final List<AbstractInstance> instances = Benchmarks.loadDiabetesDataset();
		testWithTrainingSet(instances);
	}

	private void testWithTrainingSet(final List<AbstractInstance> instances) {
		final PredictiveModelWithDataBuilderBuilder predictiveModelBuilderBuilder = new PredictiveModelWithDataBuilderBuilder(
				new RandomForestBuilderBuilder());
		final StationaryCrossValidator crossVal = new StationaryCrossValidator(4, 4,
				new LogCrossValLossFunction());
		PredictiveModelOptimizer predictiveModelOptimizer = new PredictiveModelOptimizer(
				predictiveModelBuilderBuilder, instances, crossVal);
		final Map<String, Object> optimalParameters = predictiveModelOptimizer
				.determineOptimalConfiguration();
		logger.info("Optimal parameters: " + optimalParameters);
		RandomForestBuilder defaultRFBuilder = new RandomForestBuilder();
		final PredictiveModelWithDataBuilder optimalRFBuilder = predictiveModelBuilderBuilder
				.buildBuilder(optimalParameters);
		double defaultLoss = crossVal.getCrossValidatedLoss(defaultRFBuilder, instances);
		double optimizedLoss = crossVal.getCrossValidatedLoss(optimalRFBuilder, instances);
		logger.info("Default PM loss: " + defaultLoss + ", optimized PM loss: " + optimizedLoss);
		Assert.assertTrue(optimizedLoss <= defaultLoss, "Default PM loss (" + defaultLoss
				+ ") should be higher or equal to optimized PM loss (" + optimizedLoss + ")");
	}

}
