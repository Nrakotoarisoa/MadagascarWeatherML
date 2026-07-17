using Microsoft.ML.Data;

namespace MadagascarWeatherML.Models
{
    /// <summary>
    /// Les 6 catégories de type de météo retournées par le système. L'ordre des valeurs doit
    /// correspondre exactement au mapping utilisé lors de l'encodage du label LabelTypeMeteo
    /// (KeyType count: 6) dans WeatherFeature.cs.
    /// </summary>
    public enum WeatherType
    {
        Ensoleille = 0,
        Nuageux = 1,
        Pluvieux = 2,
        Orageux = 3,
        Brouillard = 4,
        Autre = 5
    }

    /// <summary>
    /// Sortie brute du modèle de régression de température (LightGbmRegressionTrainer).
    /// ML.NET nomme par défaut la colonne de sortie "Score" ; ColumnName la rend explicite.
    /// </summary>
    public class TemperaturePredictionResult
    {
        [ColumnName("Score")]
        public float TemperaturePrevue { get; set; }
    }

    /// <summary>Sortie brute du modèle de régression de précipitation.</summary>
    public class PrecipitationPredictionResult
    {
        [ColumnName("Score")]
        public float PrecipitationPrevue { get; set; }
    }

    /// <summary>Sortie brute du modèle de régression de vitesse du vent.</summary>
    public class WindPredictionResult
    {
        [ColumnName("Score")]
        public float VitesseVentPrevue { get; set; }
    }

    /// <summary>
    /// Sortie brute du modèle de classification multiclasse du type de météo. Corrigé : le
    /// pipeline sauvegardé par WeatherTypeTrainer inclut un MapKeyToValue final, donc
    /// PredictedLabel est déjà une chaîne texte ("Pluvieux", etc.) et non une clé (uint) —
    /// contrairement à une version précédente de cette classe. Scores contient la distribution
    /// de probabilité sur les classes (son maximum sert de "confidence" dans WeatherForecastEngine).
    /// Voir aussi WeatherTypeScoringOutput dans WeatherForecastEngine.cs, structurellement identique.
    /// </summary>
    public class WeatherTypePredictionResult
    {
        [ColumnName("PredictedLabel")]
        public string TypeMeteoPrevu { get; set; }

        [ColumnName("Score")]
        public float[] Scores { get; set; }
    }

    /// <summary>
    /// DTO générique de prévision pour un jour (une famille de modèles + un horizon). Conservé pour
    /// référence/réutilisation éventuelle en dehors de l'API, mais le moteur d'inférence réel
    /// (WeatherForecastEngine.Predict) retourne désormais WeatherForecastEngine.WeatherForecastResult,
    /// dont les propriétés (Temperature, Precipitation, WindSpeed, WeatherType, Confidence) sont
    /// nommées pour correspondre exactement au contrat JSON attendu par l'API.
    /// </summary>
    public class WeatherPredictionResult
    {
        /// <summary>Date prévue (D + Horizon).</summary>
        public string Date { get; set; }

        /// <summary>Province malgache concernée par la prévision.</summary>
        public string Province { get; set; }

        /// <summary>Horizon de prévision en jours (1 à 7) ayant généré cette ligne.</summary>
        public int Horizon { get; set; }

        /// <summary>Température prévue (°C).</summary>
        public float TemperaturePrevue { get; set; }

        /// <summary>Précipitations prévues (mm).</summary>
        public float PrecipitationPrevue { get; set; }

        /// <summary>Vitesse du vent prévue (km/h).</summary>
        public float VitesseVentPrevue { get; set; }

        /// <summary>Type de météo prévu, sous forme lisible.</summary>
        public WeatherType TypeMeteoPrevu { get; set; }
    }
}
