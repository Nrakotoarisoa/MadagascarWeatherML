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
        /// Dictionnaire associant le nom de chaque province malgache (Analamanga, Fianarantsoa,
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
        /// Colonnes CSV attendues (avec en-tête, ordre fixe) :
        /// date,latitude,longitude,temperature_moyenne,temperature_min,temperature_max,
        /// humidite,pression,precipitation,vitesse_vent,direction_vent,code_meteo
        /// </summary>
        private IEnumerable<WeatherRawData> LoadSingleProvinceFile(string filePath, string province)
        {
            using var reader = new StreamReader(filePath);

            string headerLine = reader.ReadLine();
            if (headerLine == null)
                yield break;

            string line;

            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var columns = line.Split(',');

                // Ligne malformée (nombre de colonnes incorrect) : ignorée ici, sera de toute
                // façon éliminée par DataCleaner si des valeurs manquantes en résultent.
                if (columns.Length < 12)
                    continue;

                yield return new WeatherRawData
                {
                    Date = columns[0].Trim(),
                    Latitude = ParseFloatOrNaN(columns[1]),
                    Longitude = ParseFloatOrNaN(columns[2]),
                    Province = province,
                    TemperatureMoyenne = ParseFloatOrNaN(columns[3]),
                    TemperatureMinimale = ParseFloatOrNaN(columns[4]),
                    TemperatureMaximale = ParseFloatOrNaN(columns[5]),
                    Humidite = ParseFloatOrNaN(columns[6]),
                    Pression = ParseFloatOrNaN(columns[7]),
                    Precipitation = ParseFloatOrNaN(columns[8]),
                    VitesseVent = ParseFloatOrNaN(columns[9]),
                    DirectionVent = ParseFloatOrNaN(columns[10]),
                    CodeMeteo = ParseFloatOrNaN(columns[11])
                };
            }
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
