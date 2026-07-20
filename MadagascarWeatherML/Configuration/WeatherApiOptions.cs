namespace MadagascarWeatherML.Configuration;

public sealed class WeatherApiOptions
{
    public const string SectionName = "WeatherApi";
    public string ModelsDirectory { get; set; } = "Models";
    public string DataDirectory { get; set; } = "data";
    public string Version { get; set; } = "1.0.0";
}
