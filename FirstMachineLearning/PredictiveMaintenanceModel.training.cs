using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;

namespace FirstMachineLearning
{
	public partial class PredictiveMaintenanceModel
	{
		public const string RetrainFilePath = @"C:\Users\Niger\Downloads\ai4i2020.csv";
		public const char RetrainSeparatorChar = ',';
		public const bool RetrainHasHeader = true;

		private static readonly string[] inputColumnNames = new[]
		{
			@"Type",
			@"Air temperatureK",
			@"ProcessTemperatureK",
			@"RotationalSpeedRpm",
			@"TorqueNm",
			@"ToolWearMin",
			@"ProductID"
		};

		public static void Train(string outputModelPath, string inputDataFilePath = RetrainFilePath, char separatorChar = RetrainSeparatorChar, bool hasHeader = RetrainHasHeader)
		{
			MLContext mlContext = new();

			IDataView data = LoadIDataViewFromFile(mlContext, inputDataFilePath, separatorChar, hasHeader);

			ITransformer model = RetrainModel(mlContext, data);

			SaveModel(mlContext, model, data, outputModelPath);
		}

		public static IDataView LoadIDataViewFromFile(MLContext mlContext, string inputDataFilePath, char separatorChar, bool hasHeader)
		{
			return mlContext.Data.LoadFromTextFile<ModelInput>(inputDataFilePath, separatorChar, hasHeader);
		}

		public static void SaveModel(MLContext mlContext, ITransformer model, IDataView data, string modelSavePath)
		{
			DataViewSchema dataViewSchema = data.Schema;

			using FileStream fileStream = File.Create(modelSavePath);

			mlContext.Model.Save(model, dataViewSchema, fileStream);
		}

		public static ITransformer RetrainModel(MLContext mlContext, IDataView trainData)
		{
			IEstimator<ITransformer> pipeline = BuildPipeline(mlContext);

			ITransformer model = pipeline.Fit(trainData);

			return model;
		}

		public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
		{
			EstimatorChain<KeyToValueMappingTransformer> pipeline = mlContext.Transforms.Categorical.OneHotEncoding(@"Type", @"Type", outputKind: OneHotEncodingEstimator.OutputKind.Indicator)
				.Append(mlContext.Transforms.ReplaceMissingValues(new[]
				{
					new InputOutputColumnPair(@"Air temperatureK", @"Air temperatureK"),
					new InputOutputColumnPair(@"ProcessTemperatureK", @"ProcessTemperatureK"),
					new InputOutputColumnPair(@"RotationalSpeedRpm", @"RotationalSpeedRpm"),
					new InputOutputColumnPair(@"TorqueNm", @"TorqueNm"),
					new InputOutputColumnPair(@"ToolWearMin", @"ToolWearMin")
				}))
				.Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: @"ProductID", outputColumnName: @"ProductID"))
				.Append(mlContext.Transforms.Concatenate(@"Features", inputColumnNames))
				.Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"MachineFailure", inputColumnName: @"MachineFailure", addKeyValueAnnotationsAsText: false))
				.Append(mlContext.MulticlassClassification.Trainers
				.LightGbm(new LightGbmMulticlassTrainer.Options()
				{
					NumberOfLeaves = 1393,
					NumberOfIterations = 2651,
					MinimumExampleCountPerLeaf = 27,
					LearningRate = 0.999999776672986,
					LabelColumnName = @"MachineFailure",
					FeatureColumnName = @"Features",
					ExampleWeightColumnName = null,
					Booster = new GradientBooster.Options()
					{
						SubsampleFraction = 0.999999776672986,
						FeatureFraction = 0.978941618049128,
						L1Regularization = 2E-10,
						L2Regularization = 0.999999776672986
					},
					MaximumBinCountPerFeature = 233
				}))
				.Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));

			return pipeline;
		}
	}
}