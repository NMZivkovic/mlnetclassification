using Microsoft.ML.Runtime.Api;

namespace WineQualityClassification.WineQualityData
{
    public class WineQualityPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel;
    }
}
