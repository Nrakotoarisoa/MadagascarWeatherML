using System.Collections.Generic;
using System.IO;

namespace MadagascarWeatherML.Training
{
    /// <summary>
    /// Entraîne les 7 modèles de précipitation (Precipitation_J1.zip à Precipitation_J7.zip) à
    /// partir de PrecipitationTraining.csv. Pour chaque horizon, teste LightGbmRegression,
    /// FastTreeRegression et FastForestRegression, puis retient automatiquement celui avec le
    /// RMSE le plus faible.
    /// </summary>
    public class PrecipitationTrainer : RegressionTrainerBase<PrecipitationTrainingRow>
    {
        private const int HorizonMax = 7;

        /// <summary>
        /// Entraîne les 7 horizons et écrit Precipitation_J1.zip ... Precipitation_J7.zip dans
        /// modelsOutputDirectory. Peut être rappelée à tout moment avec un csvPath mis à jour
        /// pour réentraîner les modèles sur de nouvelles données.
        /// </summary>
        public List<HorizonTrainingResult> TrainAll(string csvPath, string modelsOutputDirectory)
        {
            var resultats = new List<HorizonTrainingResult>();

            for (int h = 1; h <= HorizonMax; h++)
            {
                string labelColumn = $"Precipitation_J{h}";
                string modelPath = Path.Combine(modelsOutputDirectory, $"Precipitation_J{h}.zip");

                resultats.Add(TrainHorizon(csvPath, labelColumn, modelPath));
            }

            return resultats;
        }
    }
}
