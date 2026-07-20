using Microsoft.ML.Data;

namespace MadagascarWeatherML.ML;

public sealed class WeatherTypeScoringInput
{
    [ColumnName("Province")] public string Province { get; set; } = string.Empty;
    [ColumnName("TemperatureFuture")] public float TemperatureFuture { get; set; }
    [ColumnName("PrecipitationFuture")] public float PrecipitationFuture { get; set; }
    [ColumnName("VentFutur")] public float VentFutur { get; set; }
    [ColumnName("Humidite")] public float Humidite { get; set; }
    [ColumnName("Pression")] public float Pression { get; set; }
    [ColumnName("Annee")] public float Annee { get; set; }
    [ColumnName("Mois")] public float Mois { get; set; }
    [ColumnName("Trimestre")] public float Trimestre { get; set; }
    [ColumnName("JourSemaine")] public float JourSemaine { get; set; }
    [ColumnName("JourAnnee")] public float JourAnnee { get; set; }
    [ColumnName("JourAnneeSin")] public float JourAnneeSin { get; set; }
    [ColumnName("JourAnneeCos")] public float JourAnneeCos { get; set; }
    [ColumnName("Saison")] public string Saison { get; set; } = string.Empty;
    [ColumnName("WeatherType")] public string WeatherType { get; set; } = "Autre";
}

public sealed class WeatherTypeScoringOutput
{
    [ColumnName("PredictedLabel")] public string PredictedLabel { get; set; } = string.Empty;
    [ColumnName("Score")] public float[] Score { get; set; } = Array.Empty<float>();
}
