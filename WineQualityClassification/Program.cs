using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using System;
using System.Linq;
using WineQualityClassification.Helpers;
using WineQualityClassification.Model;
using WineQualityClassification.WineQualityData;

namespace WineQualityClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            var trainingDataLocation = @"Data/winequality_white_train.csv";
            var testDataLocation = @"Data/winequality_white_test.csv";

            var modelEvaluator = new ModelEvaluator();

            var perceptronBinaryModel = new ModelBuilder(trainingDataLocation, new AveragedPerceptronBinaryClassifier()).BuildAndTrain();
            var perceptronBinaryMetrics = modelEvaluator.Evaluate(perceptronBinaryModel, testDataLocation);
            PrintMetrics("Perceptron", perceptronBinaryMetrics);

            var fastForestBinaryModel = new ModelBuilder(trainingDataLocation, new FastForestBinaryClassifier()).BuildAndTrain();
            var fastForestBinaryMetrics = modelEvaluator.Evaluate(fastForestBinaryModel, testDataLocation);
            PrintMetrics("Fast Forest Binary", fastForestBinaryMetrics);

            var fastTreeBinaryModel = new ModelBuilder(trainingDataLocation, new FastTreeBinaryClassifier()).BuildAndTrain();
            var fastTreeBinaryMetrics = modelEvaluator.Evaluate(fastTreeBinaryModel, testDataLocation);
            PrintMetrics("Fast Tree Binary", fastTreeBinaryMetrics);

            var linearSvmModel = new ModelBuilder(trainingDataLocation, new LinearSvmBinaryClassifier()).BuildAndTrain();
            var linearSvmMetrics = modelEvaluator.Evaluate(linearSvmModel, testDataLocation);
            PrintMetrics("Linear SVM", linearSvmMetrics);

            var logisticRegressionModel = new ModelBuilder(trainingDataLocation, new LogisticRegressionBinaryClassifier()).BuildAndTrain();
            var logisticRegressionMetrics = modelEvaluator.Evaluate(logisticRegressionModel, testDataLocation);
            PrintMetrics("Logistic Regression Binary", logisticRegressionMetrics);

            var sdcabModel = new ModelBuilder(trainingDataLocation, new StochasticDualCoordinateAscentBinaryClassifier()).BuildAndTrain();
            var sdcabMetrics = modelEvaluator.Evaluate(sdcabModel, testDataLocation);
            PrintMetrics("Stochastic Dual Coordinate Ascent Binary", logisticRegressionMetrics);

            VisualizeTenPredictionsForTheModel(fastForestBinaryModel, testDataLocation);

            Console.ReadLine();
        }

        private static void PrintMetrics(string name, BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name}          ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"*       Entropy: {metrics.Entropy}");
            Console.WriteLine($"*************************************************");
        }

        private static void VisualizeTenPredictionsForTheModel(
            PredictionModel<WineQualitySample, WineQualityPrediction> model,
            string testDataLocation)
        {
            var testData = new WineQualityCsvReader().GetWineQualitySamplesFromCsv(testDataLocation).ToList();
            for (int i = 0; i < 10; i++)
            {
                var prediction = model.Predict(testData[i]);
                Console.WriteLine($"-------------------------------------------------");
                Console.WriteLine($"Predicted : {prediction.PredictedLabel}");
                Console.WriteLine($"Actual:    {testData[i].Label}");
                Console.WriteLine($"-------------------------------------------------");
            }
        }
    }
}
