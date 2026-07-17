using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using MadagascarWeatherML.Models;

namespace MadagascarWeatherML.Datasets
{
    /// <summary>
    /// Construit WeatherTypeTraining.csv à partir des sorties RÉELLES des 21 modèles de régression
    /// déjà entraînés (Temperature/Precipitation/Wind_J1..J7), et non des valeurs observées.
    ///
    /// Correction importante : une version précédente utilisait les valeurs météo réellement
    /// observées comme "valeurs futures", ce qui entraînait un décalage entre l'entraînement
    /// (le classifieur voyait des valeurs parfaites) et la production (il reçoit les sorties
    /// imparfaites des modèles de régression). En scorant ici les modèles de régression déjà
    /// entraînés sur l'historique, le classifieur apprend sur exactement le même type de données
    /// — avec le même biais et le même bruit — qu'il recevra en production. C'est pour cette
    /// raison que ce générateur ne peut être exécuté qu'APRÈS l'entraînement des 21 modèles de
    /// régression (voir README.md, section "Ordre d'exécution du pipeline").
    ///
    /// Chaque horizon (1 à 7) produit ses propres lignes, car le niveau de bruit des prédictions
    /// de régression augmente avec l'horizon (J+7 est nettement moins précis que J+1) : la colonne
    /// Horizon est conservée dans le CSV pour permettre à WeatherTypeTrainer d'entraîner un
    /// classifieur réellement distinct par horizon (et non un seul modèle dupliqué 7 fois).
    /// </summary>
    public class WeatherTypeDatasetBuilder
    {
        private const int HorizonMax = 7;

        private static readonly string[] Header =
        {
            "Province", "Horizon", "TemperatureFuture", "PrecipitationFuture", "VentFutur",
            "Humidite", "Pression", "Saison", "WeatherType"
        };

        private readonly MLContext _mlContext;
        private readonly Dictionary<(string Province, DateTime Date), WeatherRawData> _rawIndex;

        private readonly Dictionary<int, PredictionEngine<WeatherFeature, TemperaturePredictionResult>> _temperatureEngines = new();
        private readonly Dictionary<int, PredictionEngine<WeatherFeature, PrecipitationPredictionResult>> _precipitationEngines = new();
        private readonly Dictionary<int, PredictionEngine<WeatherFeature, WindPredictionResult>> _windEngines = new();

        /// <param name="cleanedSortedRawData">
        /// Données brutes nettoyées, triées par Province puis Date (sortie de DataCleaner) —
        /// sert à retrouver le WeatherType réellement observé à la date cible D+h.
        /// </param>
        /// <param name="mlContext">Contexte ML.NET partagé si l'appelant en a déjà un ; sinon un nouveau est créé.</param>
        public WeatherTypeDatasetBuilder(List<WeatherRawData> cleanedSortedRawData, MLContext mlContext = null)
        {
            if (cleanedSortedRawData == null)
                throw new ArgumentNullException(nameof(cleanedSortedRawData));

            _mlContext = mlContext ?? new MLContext();
            _rawIndex = new Dictionary<(string, DateTime), WeatherRawData>();

            foreach (var row in cleanedSortedRawData)
            {
                if (TryParseDate(row.Date, out DateTime date))
                    _rawIndex[(row.Province, date.Date)] = row;
            }
        }

        /// <summary>
        /// Génère WeatherTypeTraining.csv. Doit être appelée après l'entraînement des 21 modèles
        /// de régression (fichiers Temperature/Precipitation/Wind_J1..J7.zip présents dans
        /// regressionModelsDirectory), puisque ce sont leurs sorties réelles qui alimentent ce dataset.
        /// </summary>
        /// <param name="engineeredFeatures">Sortie de WeatherFeatureEngineer (features au jour D, triées Province + Date).</param>
        /// <param name="regressionModelsDirectory">Dossier contenant les 21 modèles de régression déjà entraînés.</param>
        /// <param name="outputPath">Chemin de sortie de WeatherTypeTraining.csv.</param>
        public void BuildWeatherTypeDataset(
            List<WeatherFeature> engineeredFeatures,
            string regressionModelsDirectory,
            string outputPath)
        {
            if (engineeredFeatures == null)
                throw new ArgumentNullException(nameof(engineeredFeatures));

            var inv = CultureInfo.InvariantCulture;
            var rows = new List<string[]>();

            // engineeredFeatures est déjà trié Province + Date : on itère sans re-trier ni mélanger.
            foreach (var feature in engineeredFeatures)
            {
                if (!TryParseDate(feature.Date, out DateTime dateReference))
                    continue;

                for (int h = 1; h <= HorizonMax; h++)
                {
                    DateTime dateCible = dateReference.Date.AddDays(h);

                    // Si la date cible n'existe pas dans l'historique (trou), on saute uniquement
                    // cet horizon — les autres horizons de la même feature restent valides.
                    if (!_rawIndex.TryGetValue((feature.Province, dateCible), out var futureRow))
                        continue;

                    // Sorties RÉELLES des modèles de régression déjà entraînés pour cet horizon —
                    // exactement ce que le classifieur recevra en production via WeatherForecastEngine.
                    float temperaturePredite = ObtenirMoteurTemperature(h, regressionModelsDirectory).Predict(feature).TemperaturePrevue;
                    float precipitationPredite = ObtenirMoteurPrecipitation(h, regressionModelsDirectory).Predict(feature).PrecipitationPrevue;
                    float ventPredit = ObtenirMoteurVent(h, regressionModelsDirectory).Predict(feature).VitesseVentPrevue;

                    string saison = ResolveSaisonMalgache(dateCible.Month);
                    WeatherType typeReel = MapCodeMeteoToWeatherType(futureRow.CodeMeteo);

                    rows.Add(new[]
                    {
                        feature.Province,
                        h.ToString(inv),
                        temperaturePredite.ToString(inv),
                        precipitationPredite.ToString(inv),
                        ventPredit.ToString(inv),
                        // Humidite/Pression : mêmes valeurs "persistées" du jour de référence D
                        // que celles utilisées en inférence (WeatherForecastEngine), et non les
                        // valeurs réelles du jour cible D+h — cohérence stricte entraînement/production.
                        feature.Humidite.ToString(inv),
                        feature.Pression.ToString(inv),
                        saison,
                        typeReel.ToString()
                    });
                }
            }

            CsvDatasetWriter.Write(outputPath, Header, rows);
        }

        private PredictionEngine<WeatherFeature, TemperaturePredictionResult> ObtenirMoteurTemperature(int horizon, string modelsDirectory)
        {
            if (_temperatureEngines.TryGetValue(horizon, out var moteur))
                return moteur;

            moteur = ChargerMoteur<TemperaturePredictionResult>(modelsDirectory, "Temperature", horizon);
            _temperatureEngines[horizon] = moteur;
            return moteur;
        }

        private PredictionEngine<WeatherFeature, PrecipitationPredictionResult> ObtenirMoteurPrecipitation(int horizon, string modelsDirectory)
        {
            if (_precipitationEngines.TryGetValue(horizon, out var moteur))
                return moteur;

            moteur = ChargerMoteur<PrecipitationPredictionResult>(modelsDirectory, "Precipitation", horizon);
            _precipitationEngines[horizon] = moteur;
            return moteur;
        }

        private PredictionEngine<WeatherFeature, WindPredictionResult> ObtenirMoteurVent(int horizon, string modelsDirectory)
        {
            if (_windEngines.TryGetValue(horizon, out var moteur))
                return moteur;

            moteur = ChargerMoteur<WindPredictionResult>(modelsDirectory, "Wind", horizon);
            _windEngines[horizon] = moteur;
            return moteur;
        }

        private PredictionEngine<WeatherFeature, TDst> ChargerMoteur<TDst>(string modelsDirectory, string prefixe, int horizon)
            where TDst : class, new()
        {
            string chemin = Path.Combine(modelsDirectory, $"{prefixe}_J{horizon}.zip");

            if (!File.Exists(chemin))
                throw new FileNotFoundException(
                    $"Modèle de régression introuvable : {chemin}. " +
                    "WeatherTypeDatasetBuilder doit être exécuté après l'entraînement des modèles de régression (voir README.md).",
                    chemin);

            ITransformer modele = _mlContext.Model.Load(chemin, out _);
            return _mlContext.Model.CreatePredictionEngine<WeatherFeature, TDst>(modele);
        }

        /// <summary>
        /// Mapping des codes météo WMO fournis par Open-Meteo (colonne CodeMeteo) vers les 6
        /// catégories métier. Basé sur la nomenclature standard WMO 4677 utilisée par Open-Meteo.
        /// </summary>
        private static WeatherType MapCodeMeteoToWeatherType(float codeMeteo)
        {
            int code = (int)Math.Round(codeMeteo);

            return code switch
            {
                0 or 1 => WeatherType.Ensoleille,
                2 or 3 => WeatherType.Nuageux,
                45 or 48 => WeatherType.Brouillard,
                51 or 53 or 55 or 56 or 57 => WeatherType.Pluvieux,
                61 or 63 or 65 or 66 or 67 => WeatherType.Pluvieux,
                80 or 81 or 82 => WeatherType.Pluvieux,
                95 or 96 or 99 => WeatherType.Orageux,
                _ => WeatherType.Autre
            };
        }

        /// <summary>
        /// Découpage climatique malgache en 2 saisons, identique à celui utilisé dans
        /// WeatherFeatureEngineer et WeatherForecastEngine (cohérence indispensable).
        /// </summary>
        private static string ResolveSaisonMalgache(int mois)
        {
            bool saisonChaudeEtPluvieuse = mois >= 11 || mois <= 4;
            return saisonChaudeEtPluvieuse ? "SaisonChaudeEtPluvieuse" : "SaisonFraicheEtSeche";
        }

        private static bool TryParseDate(string rawDate, out DateTime date)
        {
            return DateTime.TryParseExact(
                rawDate, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out date)
                || DateTime.TryParse(rawDate, CultureInfo.InvariantCulture, DateTimeStyles.None, out date);
        }
    }
}
