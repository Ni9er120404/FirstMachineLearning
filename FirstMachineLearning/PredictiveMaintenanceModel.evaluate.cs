using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Immutable;

namespace FirstMachineLearning
{
	public partial class PredictiveMaintenanceModel
	{
		public static List<Tuple<string, double>> CalculatePFI(MLContext mlContext, IDataView trainData, ITransformer model, string labelColumnName)
		{
			IDataView preprocessedTrainData = model.Transform(trainData);

			ImmutableDictionary<string, MulticlassClassificationMetricsStatistics> permutationFeatureImportance =
				mlContext.MulticlassClassification
				.PermutationFeatureImportance(model, preprocessedTrainData, labelColumnName: labelColumnName);

			var featureImportanceMetrics =
				 permutationFeatureImportance
				 .Select((kvp) => new { kvp.Key, kvp.Value.MacroAccuracy })
				 .OrderByDescending(myFeatures => Math.Abs(myFeatures.MacroAccuracy.Mean));

			List<Tuple<string, double>> featurePFI = new();

			featurePFI.AddRange(from feature in featureImportanceMetrics
								let pfiValue = Math.Abs(feature.MacroAccuracy.Mean)
								select new Tuple<string, double>(feature.Key, pfiValue));
			return featurePFI;
		}
	}
}