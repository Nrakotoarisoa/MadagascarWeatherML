using System;
using System.Collections.Generic;
using System.Linq;
using MadagascarWeatherML.Models;

namespace MadagascarWeatherML.DataPreparation
{
    /// <summary>Rapport détaillé du nettoyage, utile pour la traçabilité académique du pipeline.</summary>
    public class DataCleaningReport
    {
        public int LignesInitiales { get; set; }
        public int LignesSupprimeesValeursNulles { get; set; }
        public int LignesSupprimeesDoublons { get; set; }
        public int LignesSupprimeesErreursMesure { get; set; }
        public int LignesSupprimeesAberrantes { get; set; }
        public int LignesFinales { get; set; }
    }

    /// <summary>
    /// Nettoie les données brutes Open-Meteo (valeurs nulles, doublons, erreurs de mesure et
    /// valeurs statistiquement aberrantes), puis trie le résultat par Province puis Date —
    /// ordre requis pour que WeatherFeatureEngineer calcule correctement les variables historiques.
    /// </summary>
    public class DataCleaner
    {
        // Bornes physiques réalistes pour le climat malgache (détection d'erreurs de mesure grossières)
        private const float TemperatureMin = -5f;
        private const float TemperatureMax = 45f;
        private const float HumiditeMin = 0f;
        private const float HumiditeMax = 100f;
        private const float PressionMin = 870f;
        private const float PressionMax = 1080f;
        private const float PrecipitationMin = 0f;
        private const float PrecipitationMax = 500f;   // cumul journalier extrême (cyclone)
        private const float VentMin = 0f;
        private const float VentMax = 250f;             // rafale cyclonique extrême
        private const float DirectionVentMin = 0f;
        private const float DirectionVentMax = 360f;

        // Seuil de détection des valeurs aberrantes (méthode IQR élargie, adaptée aux extrêmes météo)
        private const double IqrMultiplicateur = 3.0;
        private const int TailleMinimaleGroupePourIqr = 10;

        /// <summary>
        /// Exécute le nettoyage complet et retourne les données triées par Province puis Date,
        /// prêtes pour l'étape de feature engineering.
        /// </summary>
        public List<WeatherRawData> Clean(IEnumerable<WeatherRawData> rawData, out DataCleaningReport report)
        {
            if (rawData == null)
                throw new ArgumentNullException(nameof(rawData));

            var data = rawData.ToList();
            report = new DataCleaningReport { LignesInitiales = data.Count };

            data = RemoveNulls(data, report);
            data = RemoveDuplicates(data, report);
            data = RemoveMeasurementErrors(data, report);
            data = RemoveStatisticalOutliers(data, report);

            data = data
                .OrderBy(r => r.Province, StringComparer.OrdinalIgnoreCase)
                .ThenBy(r => r.Date, StringComparer.Ordinal) // format ISO yyyy-MM-dd => tri lexicographique = tri chronologique
                .ToList();

            report.LignesFinales = data.Count;
            return data;
        }

        /// <summary>Supprime les lignes contenant une valeur nulle, vide ou NaN sur un champ requis.</summary>
        private List<WeatherRawData> RemoveNulls(List<WeatherRawData> data, DataCleaningReport report)
        {
            int before = data.Count;

            var cleaned = data.Where(r =>
                !string.IsNullOrWhiteSpace(r.Date) &&
                !string.IsNullOrWhiteSpace(r.Province) &&
                !float.IsNaN(r.TemperatureMoyenne) &&
                !float.IsNaN(r.TemperatureMinimale) &&
                !float.IsNaN(r.TemperatureMaximale) &&
                !float.IsNaN(r.Humidite) &&
                !float.IsNaN(r.Pression) &&
                !float.IsNaN(r.Precipitation) &&
                !float.IsNaN(r.VitesseVent) &&
                !float.IsNaN(r.DirectionVent) &&
                !float.IsNaN(r.CodeMeteo)
            ).ToList();

            report.LignesSupprimeesValeursNulles = before - cleaned.Count;
            return cleaned;
        }

        /// <summary>Supprime les doublons (Province + Date identiques), en conservant la première occurrence.</summary>
        private List<WeatherRawData> RemoveDuplicates(List<WeatherRawData> data, DataCleaningReport report)
        {
            int before = data.Count;

            var cleaned = data
                .GroupBy(r => (Province: r.Province.Trim().ToLowerInvariant(), Date: r.Date.Trim()))
                .Select(g => g.First())
                .ToList();

            report.LignesSupprimeesDoublons = before - cleaned.Count;
            return cleaned;
        }

        /// <summary>Supprime les erreurs de mesure : valeurs hors des bornes physiques réalistes ou incohérentes entre elles.</summary>
        private List<WeatherRawData> RemoveMeasurementErrors(List<WeatherRawData> data, DataCleaningReport report)
        {
            int before = data.Count;

            var cleaned = data.Where(r =>
                InRange(r.TemperatureMoyenne, TemperatureMin, TemperatureMax) &&
                InRange(r.TemperatureMinimale, TemperatureMin, TemperatureMax) &&
                InRange(r.TemperatureMaximale, TemperatureMin, TemperatureMax) &&
                r.TemperatureMinimale <= r.TemperatureMoyenne &&
                r.TemperatureMoyenne <= r.TemperatureMaximale &&
                InRange(r.Humidite, HumiditeMin, HumiditeMax) &&
                InRange(r.Pression, PressionMin, PressionMax) &&
                InRange(r.Precipitation, PrecipitationMin, PrecipitationMax) &&
                InRange(r.VitesseVent, VentMin, VentMax) &&
                InRange(r.DirectionVent, DirectionVentMin, DirectionVentMax)
            ).ToList();

            report.LignesSupprimeesErreursMesure = before - cleaned.Count;
            return cleaned;
        }

        /// <summary>
        /// Détecte les valeurs statistiquement aberrantes par la méthode IQR (interquartile range),
        /// appliquée séparément par province pour température, précipitation et vent — un climat
        /// tropical (Toamasina) et un climat semi-aride (Toliara) n'ont pas la même distribution.
        /// </summary>
        private List<WeatherRawData> RemoveStatisticalOutliers(List<WeatherRawData> data, DataCleaningReport report)
        {
            int before = data.Count;
            var toRemove = new HashSet<WeatherRawData>();

            foreach (var provinceGroup in data.GroupBy(r => r.Province))
            {
                var groupList = provinceGroup.ToList();
                MarkOutliers(groupList, r => r.TemperatureMoyenne, toRemove);
                MarkOutliers(groupList, r => r.Precipitation, toRemove);
                MarkOutliers(groupList, r => r.VitesseVent, toRemove);
            }

            var cleaned = data.Where(r => !toRemove.Contains(r)).ToList();
            report.LignesSupprimeesAberrantes = before - cleaned.Count;
            return cleaned;
        }

        private void MarkOutliers(List<WeatherRawData> group, Func<WeatherRawData, float> selector, HashSet<WeatherRawData> toRemove)
        {
            if (group.Count < TailleMinimaleGroupePourIqr)
                return; // échantillon trop petit pour un calcul IQR fiable

            var sortedValues = group.Select(selector).OrderBy(v => v).ToList();

            double q1 = Percentile(sortedValues, 0.25);
            double q3 = Percentile(sortedValues, 0.75);
            double iqr = q3 - q1;
            double lowerBound = q1 - IqrMultiplicateur * iqr;
            double upperBound = q3 + IqrMultiplicateur * iqr;

            foreach (var row in group)
            {
                float value = selector(row);
                if (value < lowerBound || value > upperBound)
                    toRemove.Add(row);
            }
        }

        private static double Percentile(List<float> sortedValues, double percentile)
        {
            double index = percentile * (sortedValues.Count - 1);
            int lower = (int)Math.Floor(index);
            int upper = (int)Math.Ceiling(index);

            if (lower == upper)
                return sortedValues[lower];

            double fraction = index - lower;
            return sortedValues[lower] + fraction * (sortedValues[upper] - sortedValues[lower]);
        }

        private static bool InRange(float value, float min, float max) => value >= min && value <= max;
    }
}
