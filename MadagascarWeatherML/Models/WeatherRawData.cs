using Microsoft.ML.Data;

namespace MadagascarWeatherML.Models
{
    /// <summary>
    /// Représente une ligne brute telle qu'elle apparaît dans les fichiers CSV historiques
    /// Open-Meteo (un fichier par province). Cette classe est utilisée uniquement pour le
    /// chargement initial via TextLoader/LoadFromTextFile — aucune feature dérivée ici.
    ///
    /// Ordre des colonnes attendu dans le CSV :
    /// Date, Latitude, Longitude, Province, TempMoyenne, TempMin, TempMax,
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

        /// <summary>Nom de la province malgache (Analamanga, Fianarantsoa, Toamasina, Mahajanga, Toliara, Antsiranana).</summary>
        [LoadColumn(3)]
        [ColumnName("Province")]
        public string Province { get; set; }

        /// <summary>Température moyenne journalière (°C).</summary>
        [LoadColumn(4)]
        [ColumnName("TemperatureMoyenne")]
        public float TemperatureMoyenne { get; set; }

        /// <summary>Température minimale journalière (°C).</summary>
        [LoadColumn(5)]
        [ColumnName("TemperatureMinimale")]
        public float TemperatureMinimale { get; set; }

        /// <summary>Température maximale journalière (°C).</summary>
        [LoadColumn(6)]
        [ColumnName("TemperatureMaximale")]
        public float TemperatureMaximale { get; set; }

        /// <summary>Humidité relative moyenne (%).</summary>
        [LoadColumn(7)]
        [ColumnName("Humidite")]
        public float Humidite { get; set; }

        /// <summary>Pression atmosphérique moyenne (hPa).</summary>
        [LoadColumn(8)]
        [ColumnName("Pression")]
        public float Pression { get; set; }

        /// <summary>Cumul des précipitations journalières (mm).</summary>
        [LoadColumn(9)]
        [ColumnName("Precipitation")]
        public float Precipitation { get; set; }

        /// <summary>Vitesse moyenne du vent (km/h).</summary>
        [LoadColumn(10)]
        [ColumnName("VitesseVent")]
        public float VitesseVent { get; set; }

        /// <summary>Direction dominante du vent (degrés, 0-360).</summary>
        [LoadColumn(11)]
        [ColumnName("DirectionVent")]
        public float DirectionVent { get; set; }

        /// <summary>Code météo brut fourni par Open-Meteo (code WMO), à mapper vers les 6 catégories métier.</summary>
        [LoadColumn(12)]
        [ColumnName("CodeMeteo")]
        public float CodeMeteo { get; set; }
    }
}
