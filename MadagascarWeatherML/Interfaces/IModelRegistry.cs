using MadagascarWeatherML.DTOs;
using Microsoft.ML;

namespace MadagascarWeatherML.Interfaces;

public interface IModelRegistry
{
    ITransformer GetRequiredModel(string family, int horizon);
    ModelStatusDto GetStatus();
}
