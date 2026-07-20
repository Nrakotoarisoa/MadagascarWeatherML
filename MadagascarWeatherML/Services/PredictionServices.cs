using MadagascarWeatherML.Interfaces;
using MadagascarWeatherML.ML;
using MadagascarWeatherML.Models;
using Microsoft.ML;

namespace MadagascarWeatherML.Services;

internal sealed class LockedPredictionEngine<TSrc, TDst> where TSrc : class where TDst : class, new()
{
    private readonly PredictionEngine<TSrc, TDst> _engine;
    private readonly object _sync = new();
    public LockedPredictionEngine(MLContext mlContext, ITransformer model) => _engine = mlContext.Model.CreatePredictionEngine<TSrc, TDst>(model);
    public TDst Predict(TSrc input) { lock (_sync) return _engine.Predict(input); }
}

public abstract class PredictionServiceBase<TSrc, TDst> where TSrc : class where TDst : class, new()
{
    private readonly IModelRegistry _registry;
    private readonly MLContext _mlContext;
    private readonly string _family;
    private readonly Dictionary<int, LockedPredictionEngine<TSrc, TDst>> _engines = new();
    private readonly object _sync = new();

    protected PredictionServiceBase(IModelRegistry registry, MLContext mlContext, string family)
    {
        _registry = registry; _mlContext = mlContext; _family = family;
    }

    protected TDst PredictCore(TSrc input, int horizon)
    {
        if (horizon is < 1 or > 7) throw new ArgumentOutOfRangeException(nameof(horizon), "L'horizon doit être compris entre 1 et 7.");
        if (!_engines.TryGetValue(horizon, out var engine))
        {
            lock (_sync)
            {
                if (!_engines.TryGetValue(horizon, out engine))
                    _engines[horizon] = engine = new LockedPredictionEngine<TSrc, TDst>(_mlContext, _registry.GetRequiredModel(_family, horizon));
            }
        }
        return engine.Predict(input);
    }
}

public sealed class TemperaturePredictionService(IModelRegistry registry, MLContext mlContext)
    : PredictionServiceBase<WeatherFeature, TemperaturePredictionResult>(registry, mlContext, "Temperature"), ITemperaturePredictionService
{ public float Predict(WeatherFeature feature, int horizon) => PredictCore(feature, horizon).TemperaturePrevue; }

public sealed class PrecipitationPredictionService(IModelRegistry registry, MLContext mlContext)
    : PredictionServiceBase<WeatherFeature, PrecipitationPredictionResult>(registry, mlContext, "Precipitation"), IPrecipitationPredictionService
{ public float Predict(WeatherFeature feature, int horizon) => PredictCore(feature, horizon).PrecipitationPrevue; }

public sealed class WindPredictionService(IModelRegistry registry, MLContext mlContext)
    : PredictionServiceBase<WeatherFeature, WindPredictionResult>(registry, mlContext, "Wind"), IWindPredictionService
{ public float Predict(WeatherFeature feature, int horizon) => PredictCore(feature, horizon).VitesseVentPrevue; }

public sealed class WeatherTypePredictionService(IModelRegistry registry, MLContext mlContext)
    : PredictionServiceBase<WeatherTypeScoringInput, WeatherTypeScoringOutput>(registry, mlContext, "WeatherType"), IWeatherTypePredictionService
{
    public (string WeatherType, float Confidence) Predict(WeatherFeature feature, int horizon, DateTime targetDate, float temperature, float precipitation, float windSpeed)
    {
        var output = PredictCore(new WeatherTypeScoringInput
        {
            Province = feature.Province,
            TemperatureFuture = temperature,
            PrecipitationFuture = precipitation,
            VentFutur = windSpeed,
            Humidite = feature.Humidite,
            Pression = feature.Pression,
            Annee = targetDate.Year,
            Mois = targetDate.Month,
            Trimestre = ((targetDate.Month - 1) / 3) + 1,
            JourSemaine = (int)targetDate.DayOfWeek == 0 ? 6 : (int)targetDate.DayOfWeek - 1,
            JourAnnee = targetDate.DayOfYear,
            JourAnneeSin = (float)Math.Sin(2 * Math.PI * targetDate.DayOfYear / 365.25),
            JourAnneeCos = (float)Math.Cos(2 * Math.PI * targetDate.DayOfYear / 365.25),
            Saison = targetDate.Month >= 11 || targetDate.Month <= 4 ? "SaisonChaudeEtPluvieuse" : "SaisonFraicheEtSeche"
        }, horizon);
        return (output.PredictedLabel, output.Score is { Length: > 0 } ? output.Score.Max() : 0f);
    }
}
