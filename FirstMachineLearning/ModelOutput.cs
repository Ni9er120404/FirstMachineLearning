using Microsoft.ML.Data;
namespace FirstMachineLearning
{
	public partial class PredictiveMaintenanceModel
	{
		public class ModelOutput
		{
			[ColumnName(@"ProductID")]
			public float[] ProductID { get; set; } = null!;

			[ColumnName(@"Type")]
			public float[] Type { get; set; } = null!;

			[ColumnName(@"Air temperatureK")]
			public float AirTemperatureK { get; set; }

			[ColumnName(@"ProcessTemperatureK")]
			public float ProcessTemperatureK { get; set; }

			[ColumnName(@"RotationalSpeedRpm")]
			public float RotationalSpeedRpm { get; set; }

			[ColumnName(@"TorqueNm")]
			public float TorqueNm { get; set; }

			[ColumnName(@"ToolWearMin")]
			public float ToolWearMin { get; set; }

			[ColumnName(@"MachineFailure")]
			public uint MachineFailure { get; set; }

			[ColumnName(@"Features")]
			public float[] Features { get; set; } = null!;

			[ColumnName(@"PredictedLabel")]
			public float PredictedLabel { get; set; }

			[ColumnName(@"Score")]
			public float[] Score { get; set; } = null!;
		}
	}
}