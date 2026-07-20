using MadagascarWeatherML.DTOs;
using MadagascarWeatherML.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace MadagascarWeatherML.Controllers;

[ApiController]
[Route("api/models")]
[Produces("application/json")]
public sealed class ModelsController(IModelRegistry modelRegistry) : ControllerBase
{
    /// <summary>Retourne l'état des modèles ML.NET chargés en mémoire.</summary>
    [HttpGet("status")]
    [ProducesResponseType(typeof(ModelStatusDto), StatusCodes.Status200OK)]
    public IActionResult Status() => Ok(modelRegistry.GetStatus());
}
