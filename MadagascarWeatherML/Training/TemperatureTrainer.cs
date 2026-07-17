using System.Collections.Generic;
using System.IO;

namespace MadagascarWeatherML.Training
{
    /// <summary>
    /// Entraîne les 7 modèles de température (Temperature_J1.zip à Temperature_J7.zip) à partir
    /// de TemperatureTraining.csv. Pour chaque horizon, teste LightGbmRegression, FastTreeRegression
    /// et FastForestRegression, puis retient automatiquement celui avec le RMSE le plus faible.
    /// </summary>
    public class TemperatureTrainer : RegressionTrainerBase<TemperatureTrainingRow>
    {
        private const int HorizonMax = 7;

        /// <summary>
        /// Entraîne les 7 horizons et écrit Temperature_J1.zip ... Temperature_J7.zip dans
        /// modelsOutputDirectory. Peut être rappelée à tout moment avec un csvPath mis à jour
        /// pour réentraîner les modèles sur de nouvelles données (aucun état conservé entre appels).
        /// </summary>
        public List<HorizonTrainingResult> TrainAll(string csvPath, string modelsOutputDirectory)
        {
            var resultats = new List<HorizonTrainingResult>();

            for (int h = 1; h <= HorizonMax; h++)
            {
                string labelColumn = $"Temperature_J{h}";
                string modelPath = Path.Combine(modelsOutputDirectory, $"Temperature_J{h}.zip");

                resultats.Add(TrainHorizon(csvPath, labelColumn, modelPath));
            }

            return resultats;
        }
    }
}
