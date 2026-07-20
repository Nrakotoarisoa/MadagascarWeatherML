using MadagascarWeatherML.DTOs;
using MadagascarWeatherML.Models;

namespace MadagascarWeatherML.Interfaces;

public interface ITemperaturePredictionService { float Predict(WeatherFeature feature, int horizon); }
public interface IPrecipitationPredictionService { float Predict(WeatherFeature feature, int horizon); }
public interface IWindPredictionService { float Predict(WeatherFeature feature, int horizon); }
public interface IWeatherTypePredictionService { (string WeatherType, float Confidence) Predict(WeatherFeature feature, int horizon, DateTime targetDate, float temperature, float precipitation, float windSpeed); }
public interface IWeatherForecastService
{
    IReadOnlyList<string> GetAvailableProvinces();
    Task<WeatherForecastResponseDto> ForecastAsync(string province, DateTime startDate, int days, CancellationToken cancellationToken = default);
    Task<WeatherDayForecastDto> ForecastDayAsync(string province, DateTime date, CancellationToken cancellationToken = default);
}
