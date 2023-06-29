using Microsoft.ML.Data;
namespace FirstMachineLearning
{
	public partial class PredictiveMaintenanceModel
	{
		public class ModelInput
		{
			[LoadColumn(1)]
			[ColumnName(@"ProductID")]
			public string ProductID { get; set; } = null!;

			[LoadColumn(2)]
			[ColumnName(@"Type")]
			public string Type { get; set; } = null!;

			[LoadColumn(3)]
			[ColumnName(@"Air temperatureK")]
			public float AirTemperatureK { get; set; }

			[LoadColumn(4)]
			[ColumnName(@"ProcessTemperatureK")]
			public float ProcessTemperatureK { get; set; }

			[LoadColumn(5)]
			[ColumnName(@"RotationalSpeedRpm")]
			public float RotationalSpeedRpm { get; set; }

			[LoadColumn(6)]
			[ColumnName(@"TorqueNm")]
			public float TorqueNm { get; set; }

			[LoadColumn(7)]
			[ColumnName(@"ToolWearMin")]
			public float ToolWearMin { get; set; }

			[LoadColumn(8)]
			[ColumnName(@"MachineFailure")]
			public float MachineFailure { get; set; }

		}
	}
}
