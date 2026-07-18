using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using MadagascarWeatherML.Models;

namespace MadagascarWeatherML.DataPreparation
{
    /// <summary>
    /// Charge un ou plusieurs fichiers CSV historiques Open-Meteo — un fichier par province de
    /// Madagascar — et les fusionne en un seul IDataView ML.NET. Les exports Open-Meteo bruts ne
    /// contiennent pas la province : elle est injectée ici à partir du mapping fourni par l'appelant.
    /// </summary>
    public class OpenMeteoCsvLoader
    {
        private readonly MLContext _mlContext;

        public OpenMeteoCsvLoader(MLContext mlContext)
        {
            _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        }

        /// <summary>
        /// Charge plusieurs fichiers CSV, un par province, et retourne un IDataView consolidé
        /// (toutes provinces mélangées, non trié — le tri Province + Date est fait par l'appelant
        /// après nettoyage, via DataCleaner puis WeatherFeatureEngineer).
        /// </summary>
        /// <param name="provinceFilePaths">
        /// Dictionnaire associant le nom de chaque province malgache (Antananarivo, Fianarantsoa,
        /// Toamasina, Mahajanga, Toliara, Antsiranana) au chemin de son fichier CSV Open-Meteo.
        /// </param>
        public IDataView LoadMultipleProvinces(IDictionary<string, string> provinceFilePaths)
        {
            if (provinceFilePaths == null || provinceFilePaths.Count == 0)
                throw new ArgumentException("Au moins un fichier CSV doit être fourni.", nameof(provinceFilePaths));

            var allRows = new List<WeatherRawData>();

            foreach (var entry in provinceFilePaths)
            {
                string province = entry.Key;
                string filePath = entry.Value;

                if (!File.Exists(filePath))
                    throw new FileNotFoundException($"Fichier CSV introuvable pour la province '{province}'.", filePath);

                allRows.AddRange(LoadSingleProvinceFile(filePath, province));
            }

            return _mlContext.Data.LoadFromEnumerable(allRows);
        }

        /// <summary>
        /// Charge les lignes brutes de toutes les provinces sous forme de liste .NET, utile quand
        /// l'appelant a besoin de manipuler les données en C# avant de reconstruire un IDataView
        /// (cas du tri Province + Date et du nettoyage, effectués en amont de la conversion ML.NET).
        /// </summary>
        public List<WeatherRawData> LoadMultipleProvincesAsList(IDictionary<string, string> provinceFilePaths)
        {
            if (provinceFilePaths == null || provinceFilePaths.Count == 0)
                throw new ArgumentException("Au moins un fichier CSV doit être fourni.", nameof(provinceFilePaths));

            var allRows = new List<WeatherRawData>();

            foreach (var entry in provinceFilePaths)
            {
                string province = entry.Key;
                string filePath = entry.Value;

                if (!File.Exists(filePath))
                    throw new FileNotFoundException($"Fichier CSV introuvable pour la province '{province}'.", filePath);

                allRows.AddRange(LoadSingleProvinceFile(filePath, province));
            }

            return allRows;
        }

        /// <summary>
        /// Charge un unique fichier CSV Open-Meteo et assigne la province fournie à chaque ligne.
        /// Supporte l'ancien format applicatif (date, latitude, longitude, temperature_moyenne...)
        /// et l'export quotidien brut Open-Meteo présent dans data/ (métadonnées puis en-tête time,
        /// temperature_2m_max, weather_code, temperature_2m_min, wind_speed_10m_max...).
        /// </summary>
        private IEnumerable<WeatherRawData> LoadSingleProvinceFile(string filePath, string province)
        {
            using var reader = new StreamReader(filePath);

            string line;
            string[] header = null;

            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var candidate = line.Split(',').Select(c => c.Trim()).ToArray();
                string firstColumn = NormalizeHeader(candidate[0]);

                if (firstColumn == "date" || firstColumn == "time")
                {
                    header = candidate;
                    break;
                }
            }

            if (header == null)
                yield break;

            var columnIndex = header
                .Select((name, index) => (Name: NormalizeHeader(name), Index: index))
                .GroupBy(c => c.Name)
                .ToDictionary(g => g.Key, g => g.First().Index);

            bool legacyFormat = columnIndex.ContainsKey("temperature_moyenne");
            var metadataCoordinates = ReadCoordinatesFromMetadata(filePath);

            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var columns = line.Split(',');
                if (columns.Length < header.Length)
                    continue;

                string date = GetString(columns, columnIndex, "date", "time");
                float latitude = GetFloat(columns, columnIndex, "latitude");
                float longitude = GetFloat(columns, columnIndex, "longitude");

                if (float.IsNaN(latitude) || float.IsNaN(longitude))
                    (latitude, longitude) = metadataCoordinates;

                float temperatureMin = legacyFormat
                    ? GetFloat(columns, columnIndex, "temperature_min")
                    : GetFloat(columns, columnIndex, "temperature_2m_min");
                float temperatureMax = legacyFormat
                    ? GetFloat(columns, columnIndex, "temperature_max")
                    : GetFloat(columns, columnIndex, "temperature_2m_max");
                float temperatureMean = legacyFormat
                    ? GetFloat(columns, columnIndex, "temperature_moyenne")
                    : MoyenneSiPossible(temperatureMin, temperatureMax);

                yield return new WeatherRawData
                {
                    Date = date,
                    Latitude = latitude,
                    Longitude = longitude,
                    Province = province,
                    TemperatureMoyenne = temperatureMean,
                    TemperatureMinimale = temperatureMin,
                    TemperatureMaximale = temperatureMax,
                    // Les exports quotidiens Open-Meteo présents dans data/ ne fournissent pas
                    // humidité/pression. On utilise des valeurs de repli climatologiques neutres
                    // pour conserver l'historique au lieu de supprimer toutes les lignes.
                    Humidite = GetFloatOrDefault(columns, columnIndex, 75f, "humidite", "relative_humidity_2m_mean"),
                    Pression = GetFloatOrDefault(columns, columnIndex, 1013.25f, "pression", "surface_pressure_mean", "pressure_msl_mean"),
                    Precipitation = legacyFormat
                        ? GetFloat(columns, columnIndex, "precipitation")
                        : GetFloat(columns, columnIndex, "precipitation_sum", "rain_sum"),
                    VitesseVent = legacyFormat
                        ? GetFloat(columns, columnIndex, "vitesse_vent")
                        : GetFloat(columns, columnIndex, "wind_speed_10m_max"),
                    DirectionVent = legacyFormat
                        ? GetFloat(columns, columnIndex, "direction_vent")
                        : GetFloat(columns, columnIndex, "wind_direction_10m_dominant"),
                    CodeMeteo = legacyFormat
                        ? GetFloat(columns, columnIndex, "code_meteo")
                        : GetFloat(columns, columnIndex, "weather_code")
                };
            }
        }

        private static string GetString(string[] columns, Dictionary<string, int> columnIndex, params string[] names)
        {
            foreach (string name in names)
            {
                if (columnIndex.TryGetValue(name, out int index) && index < columns.Length)
                    return columns[index].Trim();
            }

            return string.Empty;
        }

        private static float GetFloat(string[] columns, Dictionary<string, int> columnIndex, params string[] names)
        {
            string value = GetString(columns, columnIndex, names);
            return ParseFloatOrNaN(value);
        }

        private static float GetFloatOrDefault(string[] columns, Dictionary<string, int> columnIndex, float defaultValue, params string[] names)
        {
            float value = GetFloat(columns, columnIndex, names);
            return float.IsNaN(value) ? defaultValue : value;
        }

        private static float MoyenneSiPossible(float min, float max)
        {
            return float.IsNaN(min) || float.IsNaN(max) ? float.NaN : (min + max) / 2f;
        }

        private static (float Latitude, float Longitude) ReadCoordinatesFromMetadata(string filePath)
        {
            using var reader = new StreamReader(filePath);
            string header = reader.ReadLine();
            string values = reader.ReadLine();

            if (header == null || values == null)
                return (float.NaN, float.NaN);

            var headerColumns = header.Split(',').Select(c => NormalizeHeader(c.Trim())).ToArray();
            var valueColumns = values.Split(',');
            var index = headerColumns.Select((name, i) => (name, i)).ToDictionary(x => x.name, x => x.i);

            float latitude = index.TryGetValue("latitude", out int latIndex) && latIndex < valueColumns.Length
                ? ParseFloatOrNaN(valueColumns[latIndex])
                : float.NaN;
            float longitude = index.TryGetValue("longitude", out int lonIndex) && lonIndex < valueColumns.Length
                ? ParseFloatOrNaN(valueColumns[lonIndex])
                : float.NaN;

            return (latitude, longitude);
        }

        private static string NormalizeHeader(string header)
        {
            if (string.IsNullOrWhiteSpace(header))
                return string.Empty;

            int unitStart = header.IndexOf(" (", StringComparison.Ordinal);
            string withoutUnit = unitStart >= 0 ? header[..unitStart] : header;
            return withoutUnit.Trim().ToLowerInvariant();
        }

        private static float ParseFloatOrNaN(string value)
        {
            if (value != null && float.TryParse(value.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out float result))
                return result;

            // NaN sert de marqueur de valeur manquante/invalide, détecté ensuite par DataCleaner.
            return float.NaN;
        }
    }
}
