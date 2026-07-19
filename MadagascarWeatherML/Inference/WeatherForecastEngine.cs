using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MadagascarWeatherML.Models;
using MadagascarWeatherML.DataPreparation;

namespace MadagascarWeatherML.Inference
{
    /// <summary>
    /// Ligne de résultat retournée par le moteur, au format attendu par la future API ASP.NET
    /// Core (les noms de propriétés en PascalCase se sérialisent en camelCase par défaut avec
    /// System.Text.Json dans un contrôleur ASP.NET Core : "date", "province", "temperature", etc.).
    /// </summary>
    public class WeatherForecastResult
    {
        public string Date { get; set; }
        public string Province { get; set; }
        public float Temperature { get; set; }
        public float Precipitation { get; set; }
        public float WindSpeed { get; set; }
        public string WeatherType { get; set; }
        public float Confidence { get; set; }
    }

    /// <summary>Entrée du modèle de classification au moment du scoring (features "futures" = valeurs prédites par les 3 régressions).</summary>
    internal class WeatherTypeScoringInput
    {
        [ColumnName("Province")] public string Province { get; set; }
        [ColumnName("TemperatureFuture")] public float TemperatureFuture { get; set; }
        [ColumnName("PrecipitationFuture")] public float PrecipitationFuture { get; set; }
        [ColumnName("VentFutur")] public float VentFutur { get; set; }
        [ColumnName("Humidite")] public float Humidite { get; set; }
        [ColumnName("Pression")] public float Pression { get; set; }
        [ColumnName("Annee")] public float Annee { get; set; }
        [ColumnName("Mois")] public float Mois { get; set; }
        [ColumnName("Trimestre")] public float Trimestre { get; set; }
        [ColumnName("JourSemaine")] public float JourSemaine { get; set; }
        [ColumnName("JourAnnee")] public float JourAnnee { get; set; }
        [ColumnName("JourAnneeSin")] public float JourAnneeSin { get; set; }
        [ColumnName("JourAnneeCos")] public float JourAnneeCos { get; set; }
        [ColumnName("Saison")] public string Saison { get; set; }

        // Colonne requise par le transform MapValueToKey("Label", "WeatherType") sauvegardé
        // dans le pipeline d'entraînement. Elle n'est pas utilisée par le trainer au scoring,
        // mais ML.NET exige que la colonne source existe pour construire le PredictionEngine.
        [ColumnName("WeatherType")] public string WeatherType { get; set; }
    }

    /// <summary>Sortie brute du modèle de classification (le pipeline sauvegardé reconvertit déjà la clé en texte via MapKeyToValue).</summary>
    internal class WeatherTypeScoringOutput
    {
        [ColumnName("PredictedLabel")] public string PredictedLabel { get; set; }
        [ColumnName("Score")] public float[] Score { get; set; }
    }

    /// <summary>
    /// Moteur d'inférence de bout en bout : à partir d'une province, d'une date de début et d'un
    /// nombre de jours (1 à 7), reconstruit les features à partir de l'historique, appelle les
    /// 4 familles de modèles pour chaque horizon (Temperature_Jx, Precipitation_Jx, Wind_Jx,
    /// WeatherType_Jx) et retourne la liste des prévisions au format WeatherForecastResult.
    ///
    /// Utilisation typique dans un contrôleur ASP.NET Core :
    /// <code>
    /// var resultats = _weatherForecastEngine.Predict(province: "Antananarivo", startDate: "2026-10-01", days: 7);
    /// return Ok(resultats);
    /// </code>
    ///
    /// Note production : cette classe met en cache un PredictionEngine par horizon et par famille
    /// de modèle, ce qui n'est PAS thread-safe (limitation connue de ML.NET). Pour un déploiement
    /// ASP.NET Core réel, il est recommandé d'enregistrer PredictionEnginePool&lt;TSrc,TDst&gt; en
    /// DI (un pool par famille/horizon) plutôt que d'instancier ce moteur en singleton partagé ;
    /// à défaut, instancier ce moteur en Scoped (une instance par requête HTTP) reste sûr.
    /// </summary>
    public class WeatherForecastEngine
    {
        private const int HorizonMin = 1;
        private const int HorizonMax = 7;

        private static readonly string[] ProvincesValides =
        {
            "Antananarivo", "Fianarantsoa", "Toamasina", "Mahajanga", "Toliara", "Antsiranana"
        };

        private readonly MLContext _mlContext;
        private readonly string _modelsDirectory;

        /// <summary>Index (Province, Date) -> features déjà enrichies, construit une seule fois à l'initialisation du moteur.</summary>
        private readonly Dictionary<(string Province, string Date), WeatherFeature> _featureIndex;

        private readonly Dictionary<int, PredictionEngine<WeatherFeature, TemperaturePredictionResult>> _temperatureEngines = new();
        private readonly Dictionary<int, PredictionEngine<WeatherFeature, PrecipitationPredictionResult>> _precipitationEngines = new();
        private readonly Dictionary<int, PredictionEngine<WeatherFeature, WindPredictionResult>> _windEngines = new();
        private readonly Dictionary<int, PredictionEngine<WeatherTypeScoringInput, WeatherTypeScoringOutput>> _weatherTypeEngines = new();

        /// <param name="modelsDirectory">Dossier contenant les 28 fichiers .zip (Temperature/Precipitation/Wind/WeatherType_J1..J7).</param>
        /// <param name="cleanedHistoricalData">
        /// Données historiques nettoyées et triées (sortie de DataCleaner) : c'est l'étape "charger
        /// les données historiques nécessaires" — faite une seule fois ici plutôt qu'à chaque appel
        /// de Predict, puisque les features de tous les jours sont dérivées du même historique.
        /// </param>
        /// <param name="provinceAltitudes">Altitude connue du chef-lieu de chaque province, transmise telle quelle à WeatherFeatureEngineer.</param>
        public WeatherForecastEngine(
            string modelsDirectory,
            List<WeatherRawData> cleanedHistoricalData,
            IDictionary<string, float> provinceAltitudes = null)
        {
            _modelsDirectory = modelsDirectory ?? throw new ArgumentNullException(nameof(modelsDirectory));

            if (cleanedHistoricalData == null)
                throw new ArgumentNullException(nameof(cleanedHistoricalData));

            _mlContext = new MLContext();

            // Étape 1 (chargement) + 2 (génération des features) : faites une seule fois pour tout
            // l'historique disponible, puis indexées pour un accès O(1) par (Province, Date).
            var featuresEnrichies = new WeatherFeatureEngineer().BuildFeatures(cleanedHistoricalData, provinceAltitudes);

            _featureIndex = featuresEnrichies.ToDictionary(f => (f.Province, f.Date));
        }

        /// <summary>
        /// Génère la prévision météo pour une province, à partir d'une date de début, sur un
        /// nombre de jours donné (1 à 7 inclus).
        /// </summary>
        /// <example>Predict(province: "Antananarivo", startDate: "2026-10-01", days: 7)</example>
        public List<WeatherForecastResult> Predict(string province, string startDate, int days)
        {
            if (string.IsNullOrWhiteSpace(province) || !ProvincesValides.Contains(province, StringComparer.OrdinalIgnoreCase))
                throw new ArgumentException(
                    $"Province inconnue : '{province}'. Valeurs attendues : {string.Join(", ", ProvincesValides)}.",
                    nameof(province));

            if (days < HorizonMin || days > HorizonMax)
                throw new ArgumentOutOfRangeException(nameof(days), $"days doit être compris entre {HorizonMin} et {HorizonMax}.");

            if (!DateTime.TryParseExact(startDate, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out DateTime dateDebut))
                throw new ArgumentException("startDate doit être au format yyyy-MM-dd.", nameof(startDate));

            // Jour de référence D = veille de startDate : dernier jour réellement observé, dont les
            // features (lags, climatologie, etc.) servent d'entrée à tous les horizons J+1 à J+7 —
            // exactement comme au moment de la construction des datasets d'entraînement.
            DateTime dateReference = dateDebut.AddDays(-1);
            string cleReference = dateReference.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);

            if (!_featureIndex.TryGetValue((province, cleReference), out WeatherFeature featureReference))
            {
                throw new InvalidOperationException(
                    $"Historique insuffisant pour la province '{province}' à la date de référence {cleReference} " +
                    "(il faut au moins 7 jours d'historique continu avant startDate).");
            }

            var resultats = new List<WeatherForecastResult>();

            for (int i = 0; i < days; i++)
            {
                int horizon = i + 1;
                DateTime dateCible = dateDebut.AddDays(i);

                // Étape 3 : appel des 3 modèles de régression pour cet horizon.
                float temperaturePrevue = PredireTemperature(featureReference, horizon);
                float precipitationPrevue = PredirePrecipitation(featureReference, horizon);
                float ventPrevu = PredireVent(featureReference, horizon);

                // Étape 4 : construction de l'entrée du modèle de classification à partir des
                // sorties des 3 régressions + humidité/pression/province/saison.
                var (typeMeteo, confiance) = PredireTypeMeteo(
                    featureReference, horizon, dateCible, temperaturePrevue, precipitationPrevue, ventPrevu);

                // Étape 5 : assemblage du résultat au format attendu.
                resultats.Add(new WeatherForecastResult
                {
                    Date = dateCible.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                    Province = province,
                    Temperature = (float)Math.Round(temperaturePrevue, 1),
                    Precipitation = (float)Math.Round(Math.Max(0, precipitationPrevue), 1),
                    WindSpeed = (float)Math.Round(Math.Max(0, ventPrevu), 0),
                    WeatherType = typeMeteo,
                    Confidence = (float)Math.Round(confiance, 2)
                });
            }

            return resultats;
        }

        private float PredireTemperature(WeatherFeature feature, int horizon)
        {
            var moteur = ObtenirMoteur(_temperatureEngines, "Temperature", horizon);
            return moteur.Predict(feature).TemperaturePrevue;
        }

        private float PredirePrecipitation(WeatherFeature feature, int horizon)
        {
            var moteur = ObtenirMoteur(_precipitationEngines, "Precipitation", horizon);
            return moteur.Predict(feature).PrecipitationPrevue;
        }

        private float PredireVent(WeatherFeature feature, int horizon)
        {
            var moteur = ObtenirMoteur(_windEngines, "Wind", horizon);
            return moteur.Predict(feature).VitesseVentPrevue;
        }

        private (string TypeMeteo, float Confiance) PredireTypeMeteo(
            WeatherFeature featureReference,
            int horizon,
            DateTime dateCible,
            float temperaturePrevue,
            float precipitationPrevue,
            float ventPrevu)
        {
            var entree = new WeatherTypeScoringInput
            {
                Province = featureReference.Province,
                TemperatureFuture = temperaturePrevue,
                PrecipitationFuture = precipitationPrevue,
                VentFutur = ventPrevu,
                // Humidite/Pression ne sont pas produites par les modèles de régression du système :
                // on utilise ici l'état connu au jour de référence comme approximation (persistance).
                // Une amélioration future consisterait à substituer une climatologie mensuelle pour
                // la date cible plutôt que la dernière valeur observée.
                Humidite = featureReference.Humidite,
                Pression = featureReference.Pression,
                Annee = dateCible.Year,
                Mois = dateCible.Month,
                Trimestre = ((dateCible.Month - 1) / 3) + 1,
                JourSemaine = ToLundiZeroIndex(dateCible.DayOfWeek),
                JourAnnee = dateCible.DayOfYear,
                JourAnneeSin = (float)Math.Sin(2 * Math.PI * dateCible.DayOfYear / 365.25),
                JourAnneeCos = (float)Math.Cos(2 * Math.PI * dateCible.DayOfYear / 365.25),
                Saison = ResoudreSaisonMalgache(dateCible.Month),
                WeatherType = MadagascarWeatherML.Models.WeatherType.Autre.ToString()
            };

            if (!_weatherTypeEngines.TryGetValue(horizon, out var moteur))
            {
                string chemin = Path.Combine(_modelsDirectory, $"WeatherType_J{horizon}.zip");
                ITransformer modele = _mlContext.Model.Load(chemin, out _);
                moteur = _mlContext.Model.CreatePredictionEngine<WeatherTypeScoringInput, WeatherTypeScoringOutput>(modele);
                _weatherTypeEngines[horizon] = moteur;
            }

            var sortie = moteur.Predict(entree);
            float confiance = sortie.Score is { Length: > 0 } ? sortie.Score.Max() : 0f;

            return (sortie.PredictedLabel, confiance);
        }

        private PredictionEngine<TSrc, TDst> ObtenirMoteur<TSrc, TDst>(
            Dictionary<int, PredictionEngine<TSrc, TDst>> cache,
            string prefixeFichier,
            int horizon)
            where TSrc : class
            where TDst : class, new()
        {
            if (cache.TryGetValue(horizon, out var moteurExistant))
                return moteurExistant;

            string chemin = Path.Combine(_modelsDirectory, $"{prefixeFichier}_J{horizon}.zip");

            if (!File.Exists(chemin))
                throw new FileNotFoundException($"Modèle introuvable pour l'horizon J+{horizon} : {chemin}", chemin);

            ITransformer modele = _mlContext.Model.Load(chemin, out _);
            var moteur = _mlContext.Model.CreatePredictionEngine<TSrc, TDst>(modele);

            cache[horizon] = moteur;
            return moteur;
        }

        /// <summary>
        /// Découpage climatique malgache en 2 saisons, identique à celui utilisé dans
        /// WeatherFeatureEngineer et WeatherTypeDatasetBuilder (cohérence indispensable entre
        /// l'entraînement et l'inférence).
        /// </summary>
        private static float ToLundiZeroIndex(DayOfWeek dayOfWeek)
        {
            return (int)dayOfWeek == 0 ? 6 : (int)dayOfWeek - 1;
        }

        private static string ResoudreSaisonMalgache(int mois)
        {
            bool saisonChaudeEtPluvieuse = mois >= 11 || mois <= 4;
            return saisonChaudeEtPluvieuse ? "SaisonChaudeEtPluvieuse" : "SaisonFraicheEtSeche";
        }
    }
}
