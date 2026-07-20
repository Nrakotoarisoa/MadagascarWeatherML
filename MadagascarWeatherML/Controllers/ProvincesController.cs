using MadagascarWeatherML.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace MadagascarWeatherML.Controllers;

[ApiController]
[Route("api/provinces")]
[Produces("application/json")]
public sealed class ProvincesController(IWeatherForecastService forecastService) : ControllerBase
{
    /// <summary>Liste les provinces disponibles dans l'API.</summary>
    [HttpGet]
    [ProducesResponseType(typeof(IReadOnlyList<string>), StatusCodes.Status200OK)]
    public IActionResult Get() => Ok(forecastService.GetAvailableProvinces());
}
