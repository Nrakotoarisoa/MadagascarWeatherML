using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using MadagascarWeatherML.Models;

namespace MadagascarWeatherML.DataPreparation
{
    /// <summary>
    /// Transforme les données brutes nettoyées (triées par Province + Date) en un dataset enrichi
    /// de variables temporelles et historiques, prêt à alimenter l'étape suivante du pipeline
    /// (construction du dataset multi-horizon avec labels, non traitée dans cette classe).
    ///
    /// Chaque ligne produite correspond à un jour D pour une province donnée. Les 7 premiers jours
    /// de chaque province sont exclus car ils ne disposent pas d'un historique suffisant pour
    /// calculer le lag J-7 et les agrégats glissants sur 7 jours.
    /// </summary>
    public class WeatherFeatureEngineer
    {
        private const int TailleFenetreGlissante = 7;

        /// <summary>
        /// Construit la liste des features enrichies (variables calendaires, historiques,
        /// géographiques). Ne produit aucun label ni horizon : voir NumericForecastDatasetBuilder
        /// pour la construction des CSV avec labels J+1..J+7.
        /// </summary>
        /// <param name="cleanedSortedData">Données nettoyées, déjà triées par Province puis Date (sortie de DataCleaner).</param>
        /// <param name="provinceAltitudes">Altitude connue (mètres) du chef-lieu de chaque province, si disponible.</param>
        public List<WeatherFeature> BuildFeatures(
            List<WeatherRawData> cleanedSortedData,
            IDictionary<string, float> provinceAltitudes = null)
        {
            if (cleanedSortedData == null)
                throw new ArgumentNullException(nameof(cleanedSortedData));

            var result = new List<WeatherFeature>();

            foreach (var provinceGroup in cleanedSortedData.GroupBy(r => r.Province))
            {
                // Sécurité : re-garantit l'ordre chronologique strict à l'intérieur de la province,
                // même si l'appelant a déjà trié le dataset global.
                var rows = provinceGroup.OrderBy(r => r.Date, StringComparer.Ordinal).ToList();

                var climatologie = BuildClimatologieParJourCalendaire(rows);

                for (int i = 0; i < rows.Count; i++)
                {
                    if (i < TailleFenetreGlissante)
                        continue; // historique insuffisant pour le lag J-7 et les agrégats sur 7 jours

                    var current = rows[i];

                    if (!TryParseDate(current.Date, out DateTime date))
                        continue;

                    int previousYearIndex = FromIndex(i, TailleFenetreGlissante);

                    var feature = new WeatherFeature
                    {
                        Date = current.Date,
                        Province = current.Province,
                        Latitude = current.Latitude,
                        Longitude = current.Longitude,
                        Altitude = ResolveAltitude(current.Province, provinceAltitudes),

                        // ---- Variables temporelles ----
                        Annee = date.Year,
                        Mois = date.Month,
                        Trimestre = ((date.Month - 1) / 3) + 1,
                        JourSemaine = ToLundiZeroIndex(date.DayOfWeek),
                        JourAnnee = date.DayOfYear,
                        JourAnneeSin = (float)Math.Sin(2 * Math.PI * date.DayOfYear / 365.25),
                        JourAnneeCos = (float)Math.Cos(2 * Math.PI * date.DayOfYear / 365.25),
                        Saison = ResolveSaisonMalgache(date.Month),

                        // ---- Historique température ----
                        TemperatureLag1 = rows[i - 1].TemperatureMoyenne,
                        TemperatureLag2 = rows[i - 2].TemperatureMoyenne,
                        TemperatureLag3 = rows[i - 3].TemperatureMoyenne,
                        TemperatureLag7 = rows[i - 7].TemperatureMoyenne,
                        TemperatureRollingMean7 = Average(rows, previousYearIndex, i - 1, r => r.TemperatureMoyenne),
                        TemperatureRollingStd7 = StdDev(rows, previousYearIndex, i - 1, r => r.TemperatureMoyenne),

                        // ---- Historique précipitation ----
                        PrecipitationLag1 = rows[i - 1].Precipitation,
                        PrecipitationLag3 = rows[i - 3].Precipitation,
                        PrecipitationCumul7j = Sum(rows, previousYearIndex, i - 1, r => r.Precipitation),

                        // ---- Historique vent ----
                        VentLag1 = rows[i - 1].VitesseVent,
                        VentLag3 = rows[i - 3].VitesseVent,
                        VentRollingMean7 = Average(rows, previousYearIndex, i - 1, r => r.VitesseVent),

                        // ---- État courant ----
                        Humidite = current.Humidite,
                        Pression = current.Pression
                    };

                    var (climatoTemp, climatoPrecip, climatoVent) = GetClimatologie(climatologie, date, i, rows);
                    feature.TemperatureClimatologie = climatoTemp;
                    feature.TemperatureAnomalie = current.TemperatureMoyenne - climatoTemp;
                    feature.PrecipitationClimatologie = climatoPrecip;
                    feature.VentClimatologie = climatoVent;

                    result.Add(feature);
                }
            }

            return result;
        }

        /// <summary>
        /// Index construit à partir des (mois, jour) calendaires -> observations (année, valeurs),
        /// utilisé pour calculer la climatologie ("valeur moyenne historique pour ce jour de l'année").
        /// </summary>
        private Dictionary<(int Mois, int Jour), List<(int Annee, float Temp, float Precip, float Vent)>>
            BuildClimatologieParJourCalendaire(List<WeatherRawData> rows)
        {
            var index = new Dictionary<(int, int), List<(int, float, float, float)>>();

            foreach (var row in rows)
            {
                if (!TryParseDate(row.Date, out DateTime date))
                    continue;

                var key = (date.Month, date.Day);
                if (!index.TryGetValue(key, out var list))
                {
                    list = new List<(int, float, float, float)>();
                    index[key] = list;
                }

                list.Add((date.Year, row.TemperatureMoyenne, row.Precipitation, row.VitesseVent));
            }

            return index;
        }

        /// <summary>
        /// Climatologie = moyenne historique du même jour calendaire sur les années strictement
        /// antérieures à l'année courante (évite toute fuite d'information depuis le futur).
        /// Si aucune année antérieure n'est disponible (ex. première année du jeu de données),
        /// on retombe sur la moyenne glissante sur 7 jours déjà calculée, pour éviter une valeur manquante.
        /// </summary>
        private (float Temp, float Precip, float Vent) GetClimatologie(
            Dictionary<(int Mois, int Jour), List<(int Annee, float Temp, float Precip, float Vent)>> index,
            DateTime date,
            int currentIndex,
            List<WeatherRawData> rows)
        {
            var key = (date.Month, date.Day);

            if (index.TryGetValue(key, out var observations))
            {
                var anneesAnterieures = observations.Where(o => o.Annee < date.Year).ToList();

                if (anneesAnterieures.Count > 0)
                {
                    return (
                        anneesAnterieures.Average(o => o.Temp),
                        anneesAnterieures.Average(o => o.Precip),
                        anneesAnterieures.Average(o => o.Vent)
                    );
                }
            }

            // Repli : moyenne glissante sur les 7 derniers jours disponibles.
            int fromIndex = FromIndex(currentIndex, TailleFenetreGlissante);
            return (
                Average(rows, fromIndex, currentIndex - 1, r => r.TemperatureMoyenne),
                Average(rows, fromIndex, currentIndex - 1, r => r.Precipitation),
                Average(rows, fromIndex, currentIndex - 1, r => r.VitesseVent)
            );
        }

        private static int FromIndex(int currentIndex, int windowSize) => Math.Max(0, currentIndex - windowSize);

        private static float Average(List<WeatherRawData> rows, int fromIndex, int toIndex, Func<WeatherRawData, float> selector)
        {
            if (toIndex < fromIndex) return 0f;
            return (float)rows.Skip(fromIndex).Take(toIndex - fromIndex + 1).Average(selector);
        }

        private static float Sum(List<WeatherRawData> rows, int fromIndex, int toIndex, Func<WeatherRawData, float> selector)
        {
            if (toIndex < fromIndex) return 0f;
            return rows.Skip(fromIndex).Take(toIndex - fromIndex + 1).Sum(selector);
        }

        private static float StdDev(List<WeatherRawData> rows, int fromIndex, int toIndex, Func<WeatherRawData, float> selector)
        {
            if (toIndex < fromIndex) return 0f;

            var values = rows.Skip(fromIndex).Take(toIndex - fromIndex + 1).Select(selector).ToList();
            float mean = values.Average();
            double sommeCarres = values.Sum(v => Math.Pow(v - mean, 2));
            return (float)Math.Sqrt(sommeCarres / values.Count);
        }

        /// <summary>
        /// Découpage climatique malgache en 2 saisons (plus pertinent qu'un découpage à 4 saisons
        /// occidentales) : saison chaude et pluvieuse (novembre à avril) vs saison fraîche et sèche
        /// (mai à octobre).
        /// </summary>
        private static string ResolveSaisonMalgache(int mois)
        {
            bool saisonChaudeEtPluvieuse = mois >= 11 || mois <= 4;
            return saisonChaudeEtPluvieuse ? "SaisonChaudeEtPluvieuse" : "SaisonFraicheEtSeche";
        }

        private static float ToLundiZeroIndex(DayOfWeek dayOfWeek)
        {
            // DayOfWeek .NET : Dimanche = 0 ... Samedi = 6. On ramène à Lundi = 0 ... Dimanche = 6.
            int isoIndex = (int)dayOfWeek == 0 ? 6 : (int)dayOfWeek - 1;
            return isoIndex;
        }

        private static float ResolveAltitude(string province, IDictionary<string, float> provinceAltitudes)
        {
            if (provinceAltitudes != null && provinceAltitudes.TryGetValue(province, out float altitude))
                return altitude;

            return 0f;
        }

        private static bool TryParseDate(string rawDate, out DateTime date)
        {
            return DateTime.TryParseExact(
                rawDate,
                "yyyy-MM-dd",
                CultureInfo.InvariantCulture,
                DateTimeStyles.None,
                out date)
                || DateTime.TryParse(rawDate, CultureInfo.InvariantCulture, DateTimeStyles.None, out date);
        }
    }
}
