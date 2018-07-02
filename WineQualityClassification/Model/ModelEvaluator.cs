using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using WineQualityClassification.WineQualityData;

namespace WineQualityClassification.Model
{
    public class ModelEvaluator
    {
        /// <summary>
        /// Ussing passed testing data and model, it calculates model's accuracy.
        /// </summary>
        /// <returns>Accuracy of the model.</returns>
        public BinaryClassificationMetrics Evaluate(PredictionModel<WineQualitySample, WineQualityPrediction> model, string testDataLocation)
        {
            var testData = new TextLoader(testDataLocation).CreateFrom<WineQualitySample>(useHeader: true, separator: ';');
            var metrics = new BinaryClassificationEvaluator().Evaluate(model, testData);
            return metrics;
        }
    }
}
