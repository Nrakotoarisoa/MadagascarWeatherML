using MadagascarWeatherML.Configuration;
using MadagascarWeatherML.DTOs;
using MadagascarWeatherML.Interfaces;
using Microsoft.Extensions.Options;
using Microsoft.ML;

namespace MadagascarWeatherML.Services;

public sealed class ModelRegistry : IModelRegistry
{
    private static readonly string[] Families = ["Temperature", "Precipitation", "Wind", "WeatherType"];
    private readonly Dictionary<string, ITransformer> _models = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<string> _missingModels = [];
    private readonly WeatherApiOptions _options;
    private readonly ILogger<ModelRegistry> _logger;
    private readonly MLContext _mlContext;
    private readonly DateTimeOffset _loadedAt;

    public ModelRegistry(IOptions<WeatherApiOptions> options, ILogger<ModelRegistry> logger, MLContext mlContext, IWebHostEnvironment environment)
    {
        _options = options.Value;
        _logger = logger;
        _mlContext = mlContext;
        _loadedAt = DateTimeOffset.UtcNow;
        LoadAllModels(ResolvePath(environment.ContentRootPath, _options.ModelsDirectory));
    }

    public ITransformer GetRequiredModel(string family, int horizon)
    {
        string key = BuildKey(family, horizon);
        if (_models.TryGetValue(key, out var model)) return model;
        throw new FileNotFoundException($"Le modèle requis '{key}.zip' est absent ou n'a pas pu être chargé.");
    }

    public ModelStatusDto GetStatus() => new()
    {
        LoadedModels = _models.Keys.Order().ToArray(),
        MissingModels = _missingModels.Order().ToArray(),
        Version = _options.Version,
        LoadedAt = _loadedAt,
        MemoryUsedBytes = GC.GetTotalMemory(forceFullCollection: false)
    };

    private void LoadAllModels(string modelsDirectory)
    {
        Directory.CreateDirectory(modelsDirectory);
        foreach (string family in Families)
        {
            for (int horizon = 1; horizon <= 7; horizon++)
            {
                string key = BuildKey(family, horizon);
                string path = Path.Combine(modelsDirectory, $"{key}.zip");
                if (!File.Exists(path))
                {
                    _missingModels.Add(key);
                    _logger.LogError("Modèle manquant: {ModelPath}", path);
                    continue;
                }
                try
                {
                    _models[key] = _mlContext.Model.Load(path, out _);
                    _logger.LogInformation("Modèle chargé: {ModelPath}", path);
                }
                catch (Exception ex)
                {
                    _missingModels.Add(key);
                    _logger.LogError(ex, "Impossible de charger le modèle {ModelPath}", path);
                }
            }
        }
    }

    private static string BuildKey(string family, int horizon) => $"{family}_J{horizon}";
    private static string ResolvePath(string root, string path) => Path.IsPathRooted(path) ? path : Path.Combine(root, path);
}
