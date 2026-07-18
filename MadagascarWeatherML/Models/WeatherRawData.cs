using Microsoft.ML.Data;

namespace MadagascarWeatherML.Models
{
    /// <summary>
    /// Représente une ligne brute telle qu'elle apparaît dans les fichiers CSV historiques
    /// Open-Meteo (un fichier par province). Cette classe sert de modèle objet pour
    /// OpenMeteoCsvLoader ; les attributs LoadColumn documentent le format réel du CSV à 12
    /// colonnes, mais Province n'y figure pas (OpenMeteoCsvLoader fait un parsing manuel et
    /// l'injecte lui-même — ces attributs ne sont donc pas utilisés par un TextLoader ML.NET ici).
    ///
    /// Ordre des colonnes attendu dans le CSV (avec en-tête) :
    /// Date, Latitude, Longitude, TempMoyenne, TempMin, TempMax,
    /// Humidite, Pression, Precipitation, VitesseVent, DirectionVent, CodeMeteo
    /// </summary>
    public class WeatherRawData
    {
        /// <summary>Date de l'observation, au format ISO (yyyy-MM-dd) telle que fournie par Open-Meteo.</summary>
        [LoadColumn(0)]
        [ColumnName("Date")]
        public string Date { get; set; }

        /// <summary>Latitude de la station/province (degrés décimaux).</summary>
        [LoadColumn(1)]
        [ColumnName("Latitude")]
        public float Latitude { get; set; }

        /// <summary>Longitude de la station/province (degrés décimaux).</summary>
        [LoadColumn(2)]
        [ColumnName("Longitude")]
        public float Longitude { get; set; }

        /// <summary>
        /// Nom de la province malgache (Antananarivo, Fianarantsoa, Toamasina, Mahajanga, Toliara,
        /// Antsiranana). Absent du CSV source : injecté par OpenMeteoCsvLoader, pas d'attribut LoadColumn.
        /// </summary>
        [ColumnName("Province")]
        public string Province { get; set; }

        /// <summary>Température moyenne journalière (°C).</summary>
        [LoadColumn(3)]
        [ColumnName("TemperatureMoyenne")]
        public float TemperatureMoyenne { get; set; }

        /// <summary>Température minimale journalière (°C).</summary>
        [LoadColumn(4)]
        [ColumnName("TemperatureMinimale")]
        public float TemperatureMinimale { get; set; }

        /// <summary>Température maximale journalière (°C).</summary>
        [LoadColumn(5)]
        [ColumnName("TemperatureMaximale")]
        public float TemperatureMaximale { get; set; }

        /// <summary>Humidité relative moyenne (%).</summary>
        [LoadColumn(6)]
        [ColumnName("Humidite")]
        public float Humidite { get; set; }

        /// <summary>Pression atmosphérique moyenne (hPa).</summary>
        [LoadColumn(7)]
        [ColumnName("Pression")]
        public float Pression { get; set; }

        /// <summary>Cumul des précipitations journalières (mm).</summary>
        [LoadColumn(8)]
        [ColumnName("Precipitation")]
        public float Precipitation { get; set; }

        /// <summary>Vitesse moyenne du vent (km/h).</summary>
        [LoadColumn(9)]
        [ColumnName("VitesseVent")]
        public float VitesseVent { get; set; }

        /// <summary>Direction dominante du vent (degrés, 0-360).</summary>
        [LoadColumn(10)]
        [ColumnName("DirectionVent")]
        public float DirectionVent { get; set; }

        /// <summary>Code météo brut fourni par Open-Meteo (code WMO), à mapper vers les 6 catégories métier.</summary>
        [LoadColumn(11)]
        [ColumnName("CodeMeteo")]
        public float CodeMeteo { get; set; }
    }
}
