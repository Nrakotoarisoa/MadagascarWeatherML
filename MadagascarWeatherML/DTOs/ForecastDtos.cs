using System.ComponentModel.DataAnnotations;

namespace MadagascarWeatherML.DTOs;

public sealed class WeatherDayForecastDto
{
    public string Date { get; set; } = string.Empty;
    public string Province { get; set; } = string.Empty;
    public float Temperature { get; set; }
    public float Precipitation { get; set; }
    public float WindSpeed { get; set; }
    public string WeatherType { get; set; } = string.Empty;
    public float Confidence { get; set; }
}

public sealed class WeatherForecastResponseDto
{
    public string Province { get; set; } = string.Empty;
    public IReadOnlyList<WeatherDayForecastDto> Forecast { get; set; } = Array.Empty<WeatherDayForecastDto>();
}

public sealed class ModelStatusDto
{
    public IReadOnlyList<string> LoadedModels { get; set; } = Array.Empty<string>();
    public IReadOnlyList<string> MissingModels { get; set; } = Array.Empty<string>();
    public string Version { get; set; } = string.Empty;
    public DateTimeOffset LoadedAt { get; set; }
    public long MemoryUsedBytes { get; set; }
}

public sealed class ErrorResponseDto
{
    public string Message { get; set; } = string.Empty;
}
