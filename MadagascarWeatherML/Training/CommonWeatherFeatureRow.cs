using Microsoft.ML.Data;

namespace MadagascarWeatherML.Training
{
    /// <summary>
    /// Schéma des 27 colonnes de features communes aux 3 datasets de régression
    /// (TemperatureTraining.csv, PrecipitationTraining.csv, WindTraining.csv), telles que produites
    /// par NumericForecastDatasetBuilder. Les indices LoadColumn suivent exactement l'ordre
    /// d'écriture du CSV — ne pas réordonner sans mettre à jour le générateur de dataset en amont.
    /// </summary>
    public abstract class CommonWeatherFeatureRow
    {
        [LoadColumn(0)] [ColumnName("Province")] public string Province { get; set; }

        [LoadColumn(1)] [ColumnName("TempLag1")] public float TempLag1 { get; set; }
        [LoadColumn(2)] [ColumnName("TempLag2")] public float TempLag2 { get; set; }
        [LoadColumn(3)] [ColumnName("TempLag3")] public float TempLag3 { get; set; }
        [LoadColumn(4)] [ColumnName("TempLag7")] public float TempLag7 { get; set; }
        [LoadColumn(5)] [ColumnName("TempMoyenneGlissante7j")] public float TempMoyenneGlissante7j { get; set; }
        [LoadColumn(6)] [ColumnName("TempEcartTypeGlissant7j")] public float TempEcartTypeGlissant7j { get; set; }
        [LoadColumn(7)] [ColumnName("TempClimatologie")] public float TempClimatologie { get; set; }
        [LoadColumn(8)] [ColumnName("TempAnomalie")] public float TempAnomalie { get; set; }

        [LoadColumn(9)] [ColumnName("Humidite")] public float Humidite { get; set; }
        [LoadColumn(10)] [ColumnName("Pression")] public float Pression { get; set; }

        [LoadColumn(11)] [ColumnName("VentLag1")] public float VentLag1 { get; set; }
        [LoadColumn(12)] [ColumnName("VentLag3")] public float VentLag3 { get; set; }
        [LoadColumn(13)] [ColumnName("VentMoyenneGlissante7j")] public float VentMoyenneGlissante7j { get; set; }
        [LoadColumn(14)] [ColumnName("VentClimatologie")] public float VentClimatologie { get; set; }

        [LoadColumn(15)] [ColumnName("PrecipLag1")] public float PrecipLag1 { get; set; }
        [LoadColumn(16)] [ColumnName("PrecipLag3")] public float PrecipLag3 { get; set; }
        [LoadColumn(17)] [ColumnName("PrecipCumul7j")] public float PrecipCumul7j { get; set; }
        [LoadColumn(18)] [ColumnName("PrecipClimatologie")] public float PrecipClimatologie { get; set; }

        [LoadColumn(19)] [ColumnName("Annee")] public float Annee { get; set; }
        [LoadColumn(20)] [ColumnName("Mois")] public float Mois { get; set; }
        [LoadColumn(21)] [ColumnName("Trimestre")] public float Trimestre { get; set; }
        [LoadColumn(22)] [ColumnName("JourSemaine")] public float JourSemaine { get; set; }
        [LoadColumn(23)] [ColumnName("JourAnnee")] public float JourAnnee { get; set; }
        [LoadColumn(24)] [ColumnName("JourAnneeSin")] public float JourAnneeSin { get; set; }
        [LoadColumn(25)] [ColumnName("JourAnneeCos")] public float JourAnneeCos { get; set; }
        [LoadColumn(26)] [ColumnName("Saison")] public string Saison { get; set; }
    }

    /// <summary>Ligne de TemperatureTraining.csv : 27 features communes + 7 labels Temperature_J1..J7.</summary>
    public class TemperatureTrainingRow : CommonWeatherFeatureRow
    {
        [LoadColumn(27)] [ColumnName("Temperature_J1")] public float Temperature_J1 { get; set; }
        [LoadColumn(28)] [ColumnName("Temperature_J2")] public float Temperature_J2 { get; set; }
        [LoadColumn(29)] [ColumnName("Temperature_J3")] public float Temperature_J3 { get; set; }
        [LoadColumn(30)] [ColumnName("Temperature_J4")] public float Temperature_J4 { get; set; }
        [LoadColumn(31)] [ColumnName("Temperature_J5")] public float Temperature_J5 { get; set; }
        [LoadColumn(32)] [ColumnName("Temperature_J6")] public float Temperature_J6 { get; set; }
        [LoadColumn(33)] [ColumnName("Temperature_J7")] public float Temperature_J7 { get; set; }
    }

    /// <summary>Ligne de PrecipitationTraining.csv : 27 features communes + 7 labels Precipitation_J1..J7.</summary>
    public class PrecipitationTrainingRow : CommonWeatherFeatureRow
    {
        [LoadColumn(27)] [ColumnName("Precipitation_J1")] public float Precipitation_J1 { get; set; }
        [LoadColumn(28)] [ColumnName("Precipitation_J2")] public float Precipitation_J2 { get; set; }
        [LoadColumn(29)] [ColumnName("Precipitation_J3")] public float Precipitation_J3 { get; set; }
        [LoadColumn(30)] [ColumnName("Precipitation_J4")] public float Precipitation_J4 { get; set; }
        [LoadColumn(31)] [ColumnName("Precipitation_J5")] public float Precipitation_J5 { get; set; }
        [LoadColumn(32)] [ColumnName("Precipitation_J6")] public float Precipitation_J6 { get; set; }
        [LoadColumn(33)] [ColumnName("Precipitation_J7")] public float Precipitation_J7 { get; set; }
    }

    /// <summary>Ligne de WindTraining.csv : 27 features communes + 7 labels Wind_J1..J7.</summary>
    public class WindTrainingRow : CommonWeatherFeatureRow
    {
        [LoadColumn(27)] [ColumnName("Wind_J1")] public float Wind_J1 { get; set; }
        [LoadColumn(28)] [ColumnName("Wind_J2")] public float Wind_J2 { get; set; }
        [LoadColumn(29)] [ColumnName("Wind_J3")] public float Wind_J3 { get; set; }
        [LoadColumn(30)] [ColumnName("Wind_J4")] public float Wind_J4 { get; set; }
        [LoadColumn(31)] [ColumnName("Wind_J5")] public float Wind_J5 { get; set; }
        [LoadColumn(32)] [ColumnName("Wind_J6")] public float Wind_J6 { get; set; }
        [LoadColumn(33)] [ColumnName("Wind_J7")] public float Wind_J7 { get; set; }
    }
}
