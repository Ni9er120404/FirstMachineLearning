using Microsoft.ML;
using Microsoft.ML.Data;
namespace FirstMachineLearning
{
	public partial class PredictiveMaintenanceModel
	{
		private static readonly string MLNetModelPath = Path.GetFullPath("PredictiveMaintenanceModel.mlnet");

		public static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine = new(CreatePredictEngine, true);

		private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
		{
			MLContext mlContext = new();

			ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out DataViewSchema _);

			return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
		}

		public static IOrderedEnumerable<KeyValuePair<string, float>> PredictAllLabels(ModelInput input)
		{
			PredictionEngine<ModelInput, ModelOutput> predEngine = PredictEngine.Value;

			ModelOutput result = predEngine.Predict(input);

			return GetSortedScoresWithLabels(result);
		}

		public static IOrderedEnumerable<KeyValuePair<string, float>> GetSortedScoresWithLabels(ModelOutput result)
		{
			float[] unlabeledScores = result.Score;

			IEnumerable<string> labelNames = GetLabels(result);

			Dictionary<string, float> labeledScores = new();

			for (int i = 0; i < labelNames.Count(); i++)
			{
				string labelName = labelNames.ElementAt(i);

				labeledScores.Add(labelName.ToString(), unlabeledScores[i]);
			}

			return labeledScores.OrderByDescending(c => c.Value);
		}

		private static IEnumerable<string> GetLabels(ModelOutput result)
		{
			DataViewSchema schema = PredictEngine.Value.OutputSchema;

			DataViewSchema.Column? labelColumn = schema.GetColumnOrNull("MachineFailure") ?? throw new Exception("MachineFailure column not found. Make sure the name searched for matches the name in the schema.");

			VBuffer<float> keyNames = new();

			labelColumn.Value.GetKeyValues(ref keyNames);

			return keyNames.DenseValues().Select(x => x.ToString());
		}

		public static ModelOutput Predict(ModelInput input)
		{
			PredictionEngine<ModelInput, ModelOutput> predEngine = PredictEngine.Value;

			return predEngine.Predict(input);
		}
	}
}