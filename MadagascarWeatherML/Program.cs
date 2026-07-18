using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using MadagascarWeatherML.Models;
using MadagascarWeatherML.DataPreparation;
using MadagascarWeatherML.Datasets;
using MadagascarWeatherML.Training;
using MadagascarWeatherML.Inference;

namespace MadagascarWeatherML
{
    /// <summary>
    /// Exécute l'intégralité du pipeline décrit dans README.md, section 3 : chargement, nettoyage,
    /// feature engineering, génération des datasets, entraînement des 28 modèles, puis une
    /// prédiction de démonstration à 7 jours. Les chemins data/, datasets/ et Models/ sont
    /// résolus à partir du dossier projet pour permettre l'exécution depuis la racine du dépôt
    /// avec dotnet run --project MadagascarWeatherML/MadagascarWeatherML.csproj.
    /// </summary>
    public static class Program
    {
        private static readonly string[] Provinces =
        {
            "Antananarivo", "Fianarantsoa", "Toamasina", "Mahajanga", "Toliara", "Antsiranana"
        };

        public static void Main()
        {
            string racineProjet = ResolveProjectRoot();
            string dossierDonnees = Path.Combine(racineProjet, "data");
            string dossierDatasets = Path.Combine(racineProjet, "datasets");
            string dossierModeles = Path.Combine(racineProjet, "Models");

            // --- Étapes 1-3 : chargement, nettoyage, feature engineering -----------------------
            Console.WriteLine("=== Étape 1/8 : chargement des CSV Open-Meteo ===");
            var mlContext = new MLContext(seed: 2026);
            var loader = new OpenMeteoCsvLoader(mlContext);

            var cheminsParProvince = Provinces.ToDictionary(
                p => p,
                p => Path.Combine(dossierDonnees, $"{p.ToLowerInvariant()}.csv"));

            var brut = loader.LoadMultipleProvincesAsList(cheminsParProvince);
            Console.WriteLine($"  {brut.Count} lignes brutes chargées ({Provinces.Length} provinces).");

            Console.WriteLine("=== Étape 2/8 : nettoyage ===");
            var cleaner = new DataCleaner();
            var propre = cleaner.Clean(brut, out var rapportNettoyage);
            Console.WriteLine(
                $"  {rapportNettoyage.LignesFinales} lignes conservées sur {rapportNettoyage.LignesInitiales} " +
                $"(nulles: {rapportNettoyage.LignesSupprimeesValeursNulles}, doublons: {rapportNettoyage.LignesSupprimeesDoublons}, " +
                $"erreurs de mesure: {rapportNettoyage.LignesSupprimeesErreursMesure}, aberrantes: {rapportNettoyage.LignesSupprimeesAberrantes}).");

            Console.WriteLine("=== Étape 3/8 : feature engineering ===");
            var features = new WeatherFeatureEngineer().BuildFeatures(propre);
            Console.WriteLine($"  {features.Count} lignes de features enrichies générées.");

            // --- Étape 4 : datasets de régression -----------------------------------------------
            Console.WriteLine("=== Étape 4/8 : génération des datasets de régression ===");
            var generateur = new MLDatasetGenerator();
            generateur.GenerateNumericDatasets(propre, features, dossierDatasets);
            Console.WriteLine("  TemperatureTraining.csv, PrecipitationTraining.csv, WindTraining.csv générés.");

            // --- Étape 5 : entraînement des 21 modèles de régression ---------------------------
            Console.WriteLine("=== Étape 5/8 : entraînement des modèles de régression (21 modèles) ===");

            AfficherResultatsRegression("Température",
                new TemperatureTrainer().TrainAll(
                    Path.Combine(dossierDatasets, "TemperatureTraining.csv"), dossierModeles));

            AfficherResultatsRegression("Précipitation",
                new PrecipitationTrainer().TrainAll(
                    Path.Combine(dossierDatasets, "PrecipitationTraining.csv"), dossierModeles));

            AfficherResultatsRegression("Vent",
                new WindTrainer().TrainAll(
                    Path.Combine(dossierDatasets, "WindTraining.csv"), dossierModeles));

            // --- Étape 6 : dataset de classification (dépend des modèles de l'étape 5) --------
            Console.WriteLine("=== Étape 6/8 : génération du dataset de classification ===");
            generateur.GenerateWeatherTypeDataset(propre, features, dossierModeles, dossierDatasets, mlContext);
            Console.WriteLine("  WeatherTypeTraining.csv généré à partir des sorties réelles des modèles de régression.");

            // --- Étape 7 : entraînement des 7 modèles de classification ------------------------
            Console.WriteLine("=== Étape 7/8 : entraînement des modèles de classification (7 modèles) ===");
            var resultatsClassification = new WeatherTypeTrainer().TrainAll(
                Path.Combine(dossierDatasets, "WeatherTypeTraining.csv"), dossierModeles);

            foreach (var parHorizon in resultatsClassification.GroupBy(r => r.Horizon).OrderBy(g => g.Key))
            {
                var meilleur = parHorizon
                    .OrderByDescending(r => r.MacroAccuracy)
                    .ThenByDescending(r => r.F1Macro)
                    .ThenBy(r => r.LogLoss)
                    .First();
                Console.WriteLine(
                    $"  WeatherType_J{parHorizon.Key} -> {meilleur.Algorithme} " +
                    $"(F1Macro={meilleur.F1Macro:F3}, MacroAcc={meilleur.MacroAccuracy:F3}, " +
                    $"MicroAcc={meilleur.MicroAccuracy:F3}, LogLoss={meilleur.LogLoss:F3})");
            }

            // --- Étape 8 : prédiction de démonstration à 7 jours -------------------------------
            Console.WriteLine("=== Étape 8/8 : prédiction de démonstration ===");

            string province = "Antananarivo";
            DateTime dernierJourConnu = propre
                .Where(r => r.Province == province)
                .Select(r => DateTime.ParseExact(r.Date, "yyyy-MM-dd", CultureInfo.InvariantCulture))
                .Max();

            string dateDebut = dernierJourConnu.AddDays(1).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);

            var moteur = new WeatherForecastEngine(dossierModeles, propre);
            List<WeatherForecastResult> previsions = moteur.Predict(province, dateDebut, days: 7);

            Console.WriteLine($"  Prévisions pour {province} à partir du {dateDebut} :");
            foreach (var jour in previsions)
            {
                Console.WriteLine(
                    $"    {jour.Date} | {jour.Temperature,5:F1}°C | {jour.Precipitation,5:F1}mm | " +
                    $"{jour.WindSpeed,4:F0}km/h | {jour.WeatherType,-12} | confiance {jour.Confidence:P0}");
            }

            Console.WriteLine("=== Terminé ===");
        }


        private static string ResolveProjectRoot()
        {
            string currentDirectory = Directory.GetCurrentDirectory();

            foreach (string candidate in EnumerateCandidateRoots(currentDirectory))
            {
                if (Directory.Exists(Path.Combine(candidate, "data")))
                    return candidate;

                string nestedProject = Path.Combine(candidate, "MadagascarWeatherML");
                if (Directory.Exists(Path.Combine(nestedProject, "data")))
                    return nestedProject;
            }

            string baseDirectory = AppContext.BaseDirectory;
            foreach (string candidate in EnumerateCandidateRoots(baseDirectory))
            {
                if (Directory.Exists(Path.Combine(candidate, "data")))
                    return candidate;
            }

            return currentDirectory;
        }

        private static IEnumerable<string> EnumerateCandidateRoots(string startDirectory)
        {
            var directory = new DirectoryInfo(startDirectory);

            while (directory != null)
            {
                yield return directory.FullName;
                directory = directory.Parent;
            }
        }

        private static void AfficherResultatsRegression(string cible, List<HorizonTrainingResult> resultats)
        {
            foreach (var r in resultats.OrderBy(r => r.LabelColumnName))
            {
                Console.WriteLine(
                    $"  {cible} [{r.LabelColumnName}] -> {r.AlgorithmeRetenu} " +
                    $"(RMSE={r.Rmse:F3}, MAE={r.Mae:F3}, R²={r.RSquared:F3})");
            }
        }
    }
}
