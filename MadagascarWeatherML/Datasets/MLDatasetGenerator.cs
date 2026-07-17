using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using MadagascarWeatherML.Models;

namespace MadagascarWeatherML.Datasets
{
    /// <summary>
    /// Point d'entrée de l'étape "création des datasets ML". Assemble les 4 fichiers CSV, mais en
    /// deux temps distincts car WeatherTypeTraining.csv dépend désormais des modèles de régression
    /// déjà entraînés (voir WeatherTypeDatasetBuilder pour la justification) :
    ///
    /// 1. GenerateNumericDatasets(...)   -> Temperature/Precipitation/WindTraining.csv (aucune dépendance)
    /// 2. (entraîner les 21 modèles de régression avec Temperature/Precipitation/WindTrainer)
    /// 3. GenerateWeatherTypeDataset(...) -> WeatherTypeTraining.csv (dépend des modèles de l'étape 2)
    ///
    /// Voir README.md, section "Ordre d'exécution du pipeline", pour la séquence complète.
    /// </summary>
    public class MLDatasetGenerator
    {
        /// <summary>Génère les 3 CSV de régression. Ne fait aucun nettoyage ni feature engineering supplémentaire.</summary>
        /// <param name="cleanedSortedRawData">Sortie de DataCleaner : triée par Province puis Date.</param>
        /// <param name="engineeredFeatures">Sortie de WeatherFeatureEngineer : triée par Province puis Date.</param>
        /// <param name="outputDirectory">Dossier de destination des 3 CSV.</param>
        public void GenerateNumericDatasets(
            List<WeatherRawData> cleanedSortedRawData,
            List<WeatherFeature> engineeredFeatures,
            string outputDirectory)
        {
            var numericBuilder = new NumericForecastDatasetBuilder(cleanedSortedRawData);

            numericBuilder.BuildTemperatureDataset(
                engineeredFeatures, Path.Combine(outputDirectory, "TemperatureTraining.csv"));

            numericBuilder.BuildPrecipitationDataset(
                engineeredFeatures, Path.Combine(outputDirectory, "PrecipitationTraining.csv"));

            numericBuilder.BuildWindDataset(
                engineeredFeatures, Path.Combine(outputDirectory, "WindTraining.csv"));
        }

        /// <summary>
        /// Génère WeatherTypeTraining.csv à partir des sorties réelles des 21 modèles de régression.
        /// Ne doit être appelée qu'après TemperatureTrainer/PrecipitationTrainer/WindTrainer.TrainAll().
        /// </summary>
        /// <param name="cleanedSortedRawData">Sortie de DataCleaner : triée par Province puis Date.</param>
        /// <param name="engineeredFeatures">Sortie de WeatherFeatureEngineer : triée par Province puis Date.</param>
        /// <param name="regressionModelsDirectory">Dossier contenant les 21 modèles de régression déjà entraînés.</param>
        /// <param name="outputDirectory">Dossier de destination de WeatherTypeTraining.csv.</param>
        public void GenerateWeatherTypeDataset(
            List<WeatherRawData> cleanedSortedRawData,
            List<WeatherFeature> engineeredFeatures,
            string regressionModelsDirectory,
            string outputDirectory,
            MLContext mlContext = null)
        {
            var weatherTypeBuilder = new WeatherTypeDatasetBuilder(cleanedSortedRawData, mlContext);

            weatherTypeBuilder.BuildWeatherTypeDataset(
                engineeredFeatures, regressionModelsDirectory, Path.Combine(outputDirectory, "WeatherTypeTraining.csv"));
        }
    }
}
