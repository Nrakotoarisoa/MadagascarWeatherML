using MadagascarWeatherML.DTOs;
using MadagascarWeatherML.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace MadagascarWeatherML.Controllers;

[ApiController]
[Route("api/weather")]
[Produces("application/json")]
public sealed class WeatherController(IWeatherForecastService forecastService, ILogger<WeatherController> logger) : ControllerBase
{
    /// <summary>Retourne une prévision météo de 1 à 7 jours pour une province.</summary>
    /// <example>GET /api/weather/forecast?province=Antananarivo&amp;startDate=2026-10-01&amp;days=7</example>
    [HttpGet("forecast")]
    [ProducesResponseType(typeof(WeatherForecastResponseDto), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponseDto), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponseDto), StatusCodes.Status404NotFound)]
    [ProducesResponseType(typeof(ErrorResponseDto), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Forecast([FromQuery] string province, [FromQuery] DateTime startDate, [FromQuery] int days = 7, CancellationToken cancellationToken = default)
        => await Execute(() => forecastService.ForecastAsync(province, startDate, days, cancellationToken));

    /// <summary>Retourne la prévision météo d'un seul jour pour une province.</summary>
    /// <example>GET /api/weather/day?province=Antananarivo&amp;date=2026-10-01</example>
    [HttpGet("day")]
    [ProducesResponseType(typeof(WeatherDayForecastDto), StatusCodes.Status200OK)]
    public async Task<IActionResult> Day([FromQuery] string province, [FromQuery] DateTime date, CancellationToken cancellationToken = default)
        => await Execute(() => forecastService.ForecastDayAsync(province, date, cancellationToken));

    private async Task<IActionResult> Execute<T>(Func<Task<T>> action)
    {
        try { return Ok(await action()); }
        catch (ArgumentException ex) { return BadRequest(new ErrorResponseDto { Message = ex.Message }); }
        catch (KeyNotFoundException ex) { return NotFound(new ErrorResponseDto { Message = ex.Message }); }
        catch (FileNotFoundException ex) { return StatusCode(500, new ErrorResponseDto { Message = ex.Message }); }
        catch (Exception ex) { logger.LogError(ex, "Erreur pendant la prédiction météo."); return StatusCode(500, new ErrorResponseDto { Message = "Erreur interne pendant la prédiction météo." }); }
    }
}
