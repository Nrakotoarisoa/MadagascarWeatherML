using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using MadagascarWeatherML.Models;

namespace MadagascarWeatherML.Datasets
{
    /// <summary>
    /// Construit les 3 datasets de régression multi-horizon (température, précipitation, vent)
    /// au format "large" : une ligne par (province, date D), avec les features déjà calculées par
    /// WeatherFeatureEngineer (donc strictement issues de données ≤ D) et 7 colonnes de label
    /// (J+1 à J+7) correspondant aux valeurs réellement observées à D+1 ... D+7.
    ///
    /// Règle stricte : si l'une des 7 dates futures (D+1 ... D+7) est absente du jeu de données
    /// (trou dans la série), la ligne entière est rejetée plutôt que d'être partiellement remplie
    /// ou approximée — cela garantit qu'aucune donnée future n'est jamais mélangée à tort avec une
    /// mauvaise date, et qu'aucun label n'est fabriqué.
    /// </summary>
    public class NumericForecastDatasetBuilder
    {
        private const int HorizonMax = 7;

        private readonly Dictionary<(string Province, DateTime Date), WeatherRawData> _rawIndex;

        /// <param name="cleanedSortedRawData">
        /// Données brutes nettoyées, triées par Province puis Date (sortie de DataCleaner) —
        /// sert uniquement à retrouver la valeur réelle observée aux dates futures D+1..D+7.
        /// </param>
        public NumericForecastDatasetBuilder(List<WeatherRawData> cleanedSortedRawData)
        {
            if (cleanedSortedRawData == null)
                throw new ArgumentNullException(nameof(cleanedSortedRawData));

            _rawIndex = new Dictionary<(string, DateTime), WeatherRawData>();

            foreach (var row in cleanedSortedRawData)
            {
                if (TryParseDate(row.Date, out DateTime date))
                    _rawIndex[(row.Province, date.Date)] = row;
            }
        }

        /// <summary>Génère TemperatureTraining.csv (labels Temperature_J1 à Temperature_J7).</summary>
        public void BuildTemperatureDataset(List<WeatherFeature> features, string outputPath)
            => BuildDataset(features, r => r.TemperatureMoyenne, "Temperature", outputPath);

        /// <summary>Génère PrecipitationTraining.csv (labels Precipitation_J1 à Precipitation_J7).</summary>
        public void BuildPrecipitationDataset(List<WeatherFeature> features, string outputPath)
            => BuildDataset(features, r => r.Precipitation, "Precipitation", outputPath);

        /// <summary>Génère WindTraining.csv (labels Wind_J1 à Wind_J7).</summary>
        public void BuildWindDataset(List<WeatherFeature> features, string outputPath)
            => BuildDataset(features, r => r.VitesseVent, "Wind", outputPath);

        private void BuildDataset(
            List<WeatherFeature> features,
            Func<WeatherRawData, float> targetSelector,
            string labelPrefix,
            string outputPath)
        {
            if (features == null)
                throw new ArgumentNullException(nameof(features));

            var header = BuildFeatureHeader()
                .Concat(Enumerable.Range(1, HorizonMax).Select(h => $"{labelPrefix}_J{h}"))
                .ToList();

            var rows = new List<string[]>();

            // features est déjà trié par Province puis Date (sortie de WeatherFeatureEngineer) :
            // on itère sans jamais re-trier ni mélanger, pour préserver l'ordre chronologique strict.
            foreach (var feature in features)
            {
                if (!TryParseDate(feature.Date, out DateTime date))
                    continue;

                var labelValues = new float[HorizonMax];
                bool horizonComplet = true;

                for (int h = 1; h <= HorizonMax; h++)
                {
                    DateTime dateCible = date.Date.AddDays(h);

                    // Recherche stricte par (Province, DateCible) : garantit qu'on ne récupère
                    // jamais une valeur d'une autre province ou d'un mauvais décalage temporel.
                    if (!_rawIndex.TryGetValue((feature.Province, dateCible), out var futureRow))
                    {
                        horizonComplet = false; // trou dans la série : ligne rejetée, aucune approximation
                        break;
                    }

                    labelValues[h - 1] = targetSelector(futureRow);
                }

                if (!horizonComplet)
                    continue;

                var row = BuildFeatureValues(feature)
                    .Concat(labelValues.Select(v => v.ToString(CultureInfo.InvariantCulture)))
                    .ToArray();

                rows.Add(row);
            }

            CsvDatasetWriter.Write(outputPath, header, rows);
        }

        /// <summary>
        /// En-tête des features communes aux 3 datasets de régression : province, historique
        /// température, humidité, pression, historique vent, historique précipitation, variables
        /// temporelles — exactement l'ensemble demandé, déjà produit par WeatherFeatureEngineer.
        /// </summary>
        private static List<string> BuildFeatureHeader() => new()
        {
            "Province",
            // Historique température
            "TempLag1", "TempLag2", "TempLag3", "TempLag7",
            "TempMoyenneGlissante7j", "TempEcartTypeGlissant7j",
            "TempClimatologie", "TempAnomalie",
            // Humidité / Pression (état courant au jour D)
            "Humidite", "Pression",
            // Historique vent
            "VentLag1", "VentLag3", "VentMoyenneGlissante7j", "VentClimatologie",
            // Historique précipitation
            "PrecipLag1", "PrecipLag3", "PrecipCumul7j", "PrecipClimatologie",
            // Variables temporelles
            "Annee", "Mois", "Trimestre", "JourSemaine", "JourAnnee",
            "JourAnneeSin", "JourAnneeCos", "Saison"
        };

        private static IEnumerable<string> BuildFeatureValues(WeatherFeature f)
        {
            var inv = CultureInfo.InvariantCulture;
            return new[]
            {
                f.Province,
                f.TemperatureLag1.ToString(inv), f.TemperatureLag2.ToString(inv),
                f.TemperatureLag3.ToString(inv), f.TemperatureLag7.ToString(inv),
                f.TemperatureRollingMean7.ToString(inv), f.TemperatureRollingStd7.ToString(inv),
                f.TemperatureClimatologie.ToString(inv), f.TemperatureAnomalie.ToString(inv),
                f.Humidite.ToString(inv), f.Pression.ToString(inv),
                f.VentLag1.ToString(inv), f.VentLag3.ToString(inv),
                f.VentRollingMean7.ToString(inv), f.VentClimatologie.ToString(inv),
                f.PrecipitationLag1.ToString(inv), f.PrecipitationLag3.ToString(inv),
                f.PrecipitationCumul7j.ToString(inv), f.PrecipitationClimatologie.ToString(inv),
                f.Annee.ToString(inv), f.Mois.ToString(inv), f.Trimestre.ToString(inv),
                f.JourSemaine.ToString(inv), f.JourAnnee.ToString(inv),
                f.JourAnneeSin.ToString(inv), f.JourAnneeCos.ToString(inv),
                f.Saison
            };
        }

        private static bool TryParseDate(string rawDate, out DateTime date)
        {
            return DateTime.TryParseExact(
                rawDate, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out date)
                || DateTime.TryParse(rawDate, CultureInfo.InvariantCulture, DateTimeStyles.None, out date);
        }
    }
}
