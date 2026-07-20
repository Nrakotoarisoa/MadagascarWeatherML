using MadagascarWeatherML.Configuration;
using MadagascarWeatherML.Interfaces;
using MadagascarWeatherML.Services;
using Microsoft.OpenApi.Models;
using Microsoft.ML;

namespace MadagascarWeatherML.Extensions;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddWeatherApi(this IServiceCollection services, IConfiguration configuration)
    {
        services.Configure<WeatherApiOptions>(configuration.GetSection(WeatherApiOptions.SectionName));
        services.AddSingleton(new MLContext(seed: 2026));
        services.AddSingleton<IModelRegistry, ModelRegistry>();
        services.AddSingleton<ITemperaturePredictionService, TemperaturePredictionService>();
        services.AddSingleton<IPrecipitationPredictionService, PrecipitationPredictionService>();
        services.AddSingleton<IWindPredictionService, WindPredictionService>();
        services.AddSingleton<IWeatherTypePredictionService, WeatherTypePredictionService>();
        services.AddSingleton<IWeatherForecastService, WeatherForecastService>();
        return services;
    }

    public static IServiceCollection AddWeatherSwagger(this IServiceCollection services)
    {
        services.AddEndpointsApiExplorer();
        services.AddSwaggerGen(options =>
        {
            options.SwaggerDoc("v1", new OpenApiInfo
            {
                Title = "Madagascar Weather ML API",
                Version = "v1",
                Description = "API REST ASP.NET Core pour consommer des modèles ML.NET météo déjà entraînés."
            });
            var xml = $"{typeof(Program).Assembly.GetName().Name}.xml";
            var xmlPath = Path.Combine(AppContext.BaseDirectory, xml);
            if (File.Exists(xmlPath)) options.IncludeXmlComments(xmlPath);
        });
        return services;
    }
}
