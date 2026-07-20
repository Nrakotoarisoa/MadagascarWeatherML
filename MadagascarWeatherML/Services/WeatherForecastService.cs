using System.Globalization;
using MadagascarWeatherML.Configuration;
using MadagascarWeatherML.DataPreparation;
using MadagascarWeatherML.DTOs;
using MadagascarWeatherML.Interfaces;
using MadagascarWeatherML.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML;

namespace MadagascarWeatherML.Services;

public sealed class WeatherForecastService : IWeatherForecastService
{
    private static readonly string[] Provinces = ["Antananarivo", "Fianarantsoa", "Toamasina", "Mahajanga", "Toliara", "Antsiranana"];
    private readonly Dictionary<(string Province, string Date), WeatherFeature> _features;
    private readonly ITemperaturePredictionService _temperature;
    private readonly IPrecipitationPredictionService _precipitation;
    private readonly IWindPredictionService _wind;
    private readonly IWeatherTypePredictionService _weatherType;
    private readonly ILogger<WeatherForecastService> _logger;

    public WeatherForecastService(IOptions<WeatherApiOptions> options, IWebHostEnvironment environment, MLContext mlContext,
        ITemperaturePredictionService temperature, IPrecipitationPredictionService precipitation, IWindPredictionService wind,
        IWeatherTypePredictionService weatherType, ILogger<WeatherForecastService> logger)
    {
        _temperature = temperature; _precipitation = precipitation; _wind = wind; _weatherType = weatherType; _logger = logger;
        string dataDirectory = ResolvePath(environment.ContentRootPath, options.Value.DataDirectory);
        var paths = Provinces.ToDictionary(p => p, p => Path.Combine(dataDirectory, $"{p.ToLowerInvariant()}.csv"));
        var raw = new OpenMeteoCsvLoader(mlContext).LoadMultipleProvincesAsList(paths);
        var cleaned = new DataCleaner().Clean(raw, out _);
        _features = new WeatherFeatureEngineer().BuildFeatures(cleaned).ToDictionary(f => (f.Province, f.Date));
    }

    public IReadOnlyList<string> GetAvailableProvinces() => Provinces;

    public Task<WeatherForecastResponseDto> ForecastAsync(string province, DateTime startDate, int days, CancellationToken cancellationToken = default)
    {
        Validate(province, days);
        string canonicalProvince = Provinces.First(p => p.Equals(province, StringComparison.OrdinalIgnoreCase));
        string referenceDate = startDate.AddDays(-1).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        if (!_features.TryGetValue((canonicalProvince, referenceDate), out var feature))
            throw new KeyNotFoundException($"Historique insuffisant pour '{canonicalProvince}' à la date de référence {referenceDate}.");

        var forecasts = new List<WeatherDayForecastDto>(days);
        for (int i = 0; i < days; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int horizon = i + 1;
            DateTime target = startDate.AddDays(i);
            float temp = _temperature.Predict(feature, horizon);
            float precip = _precipitation.Predict(feature, horizon);
            float wind = _wind.Predict(feature, horizon);
            var (type, confidence) = _weatherType.Predict(feature, horizon, target, temp, precip, wind);
            forecasts.Add(new WeatherDayForecastDto
            {
                Date = target.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture), Province = canonicalProvince,
                Temperature = (float)Math.Round(temp, 1), Precipitation = (float)Math.Round(Math.Max(0, precip), 1),
                WindSpeed = (float)Math.Round(Math.Max(0, wind), 1), WeatherType = type,
                Confidence = (float)Math.Round(confidence, 2)
            });
        }
        return Task.FromResult(new WeatherForecastResponseDto { Province = canonicalProvince, Forecast = forecasts });
    }

    public async Task<WeatherDayForecastDto> ForecastDayAsync(string province, DateTime date, CancellationToken cancellationToken = default)
        => (await ForecastAsync(province, date, 1, cancellationToken)).Forecast[0];

    private static void Validate(string province, int days)
    {
        if (string.IsNullOrWhiteSpace(province)) throw new ArgumentException("Le paramètre province est obligatoire.", nameof(province));
        if (!Provinces.Contains(province, StringComparer.OrdinalIgnoreCase)) throw new KeyNotFoundException($"Province inconnue: {province}.");
        if (days is < 1 or > 7) throw new ArgumentOutOfRangeException(nameof(days), "Le nombre de jours doit être compris entre 1 et 7.");
    }

    private static string ResolvePath(string root, string path) => Path.IsPathRooted(path) ? path : Path.Combine(root, path);
}
