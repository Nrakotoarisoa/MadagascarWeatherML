using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MadagascarWeatherML.Training
{
    /// <summary>Résultat d'entraînement pour un horizon (J+1 à J+7) d'une cible donnée.</summary>
    public class HorizonTrainingResult
    {
        public string LabelColumnName { get; set; }
        public string AlgorithmeRetenu { get; set; }
        public double Rmse { get; set; }
        public double Mae { get; set; }
        public double RSquared { get; set; }
        public string ModelPath { get; set; }

        /// <summary>Détail RMSE/MAE/R² des 3 algorithmes testés, pour traçabilité académique.</summary>
        public List<(string Algorithme, double Rmse, double Mae, double RSquared)> ComparaisonAlgorithmes { get; set; }
    }

    /// <summary>
    /// Base commune aux 3 entraîneurs de régression (température, précipitation, vent). Pour un
    /// horizon donné : charge le CSV, sépare train/test de façon strictement chronologique (par
    /// province), entraîne 3 algorithmes candidats (LightGbm, FastTree, FastForest), les évalue
    /// (RMSE, MAE, R²), sélectionne automatiquement le meilleur, puis le réentraîne sur l'ensemble
    /// des données disponibles avant de sauvegarder le modèle final au format .zip.
    ///
    /// Conçue pour être réutilisée pour un réentraînement futur : il suffit de rappeler TrainHorizon
    /// (ou TrainAll dans les sous-classes) avec un CSV mis à jour — aucun état n'est conservé entre deux appels.
    /// </summary>
    /// <typeparam name="TRow">Classe de schéma du CSV (doit hériter de CommonWeatherFeatureRow).</typeparam>
    public abstract class RegressionTrainerBase<TRow> where TRow : CommonWeatherFeatureRow, new()
    {
        private const double FractionEntrainement = 0.8;

        protected readonly MLContext MlContext;

        protected RegressionTrainerBase(int seed = 2026)
        {
            MlContext = new MLContext(seed: seed);
        }

        /// <summary>
        /// Entraîne, évalue et sélectionne automatiquement le meilleur modèle pour un horizon donné,
        /// puis sauvegarde le modèle final réentraîné sur toutes les données disponibles.
        /// </summary>
        /// <param name="csvPath">Chemin du CSV d'entraînement (Temperature/Precipitation/WindTraining.csv).</param>
        /// <param name="labelColumnName">Colonne label à prédire pour cet horizon (ex. "Temperature_J3").</param>
        /// <param name="outputModelPath">Chemin de sortie du modèle .zip (ex. Models/Temperature_J3.zip).</param>
        public HorizonTrainingResult TrainHorizon(string csvPath, string labelColumnName, string outputModelPath)
        {
            if (!File.Exists(csvPath))
                throw new FileNotFoundException("Dataset d'entraînement introuvable.", csvPath);

            IDataView fullData = MlContext.Data.LoadFromTextFile<TRow>(csvPath, hasHeader: true, separatorChar: ',');

            var (trainData, testData) = SplitChronologiquementParProvince(fullData);

            var candidats = new (string Nom, IEstimator<ITransformer> Pipeline)[]
            {
                ("LightGbm", BuildPipeline(MlContext.Regression.Trainers.LightGbm(
                    labelColumnName: labelColumnName, featureColumnName: "Features"))),

                ("FastTree", BuildPipeline(MlContext.Regression.Trainers.FastTree(
                    labelColumnName: labelColumnName, featureColumnName: "Features"))),

                ("FastForest", BuildPipeline(MlContext.Regression.Trainers.FastForest(
                    labelColumnName: labelColumnName, featureColumnName: "Features")))
            };

            var evaluations = new List<(string Nom, RegressionMetrics Metrics)>();

            foreach (var (nom, pipeline) in candidats)
            {
                ITransformer modele = pipeline.Fit(trainData);
                IDataView predictions = modele.Transform(testData);
                RegressionMetrics metrics = MlContext.Regression.Evaluate(
                    predictions, labelColumnName: labelColumnName, scoreColumnName: "Score");

                evaluations.Add((nom, metrics));
            }

            // Sélection automatique : RMSE le plus faible retenu comme critère principal
            // (métrique standard pour la comparaison de modèles de régression météo).
            var meilleur = evaluations.OrderBy(e => e.Metrics.RootMeanSquaredError).First();
            var pipelineRetenu = candidats.First(c => c.Nom == meilleur.Nom).Pipeline;

            // Réentraînement final du meilleur algorithme sur l'intégralité des données
            // disponibles (train + test) pour maximiser l'usage de l'historique en production.
            ITransformer modeleFinal = pipelineRetenu.Fit(fullData);

            string dossierSortie = Path.GetDirectoryName(outputModelPath);
            if (!string.IsNullOrEmpty(dossierSortie))
                Directory.CreateDirectory(dossierSortie);

            MlContext.Model.Save(modeleFinal, fullData.Schema, outputModelPath);

            return new HorizonTrainingResult
            {
                LabelColumnName = labelColumnName,
                AlgorithmeRetenu = meilleur.Nom,
                Rmse = meilleur.Metrics.RootMeanSquaredError,
                Mae = meilleur.Metrics.MeanAbsoluteError,
                RSquared = meilleur.Metrics.RSquared,
                ModelPath = outputModelPath,
                ComparaisonAlgorithmes = evaluations
                    .Select(e => (e.Nom, e.Metrics.RootMeanSquaredError, e.Metrics.MeanAbsoluteError, e.Metrics.RSquared))
                    .ToList()
            };
        }

        /// <summary>
        /// Pipeline de préprocessing commun aux 3 cibles : encodage catégoriel de Province et
        /// Saison, puis concaténation de toutes les features numériques et encodées en un seul
        /// vecteur "Features", avant d'enchaîner sur le trainer fourni.
        /// </summary>
        private IEstimator<ITransformer> BuildPipeline(IEstimator<ITransformer> trainer)
        {
            var preprocessing = MlContext.Transforms.Categorical.OneHotEncoding("ProvinceEncoded", "Province")
                .Append(MlContext.Transforms.Categorical.OneHotEncoding("SaisonEncoded", "Saison"))
                .Append(MlContext.Transforms.Concatenate("Features",
                    "ProvinceEncoded", "SaisonEncoded",
                    "TempLag1", "TempLag2", "TempLag3", "TempLag7",
                    "TempMoyenneGlissante7j", "TempEcartTypeGlissant7j",
                    "TempClimatologie", "TempAnomalie",
                    "Humidite", "Pression",
                    "VentLag1", "VentLag3", "VentMoyenneGlissante7j", "VentClimatologie",
                    "PrecipLag1", "PrecipLag3", "PrecipCumul7j", "PrecipClimatologie",
                    "Annee", "Mois", "Trimestre", "JourSemaine", "JourAnnee",
                    "JourAnneeSin", "JourAnneeCos"));

            return preprocessing.Append(trainer);
        }

        /// <summary>
        /// Sépare train/test de façon strictement chronologique, indépendamment pour chaque
        /// province : les FractionEntrainement (80%) premiers jours de chaque province vont à
        /// l'entraînement, les plus récents (20%) au test. Un split global naïf sur l'ordre brut
        /// du fichier mélangerait les provinces (le CSV est trié Province puis Date) ; ce découpage
        /// par groupe garantit qu'aucune province n'est absente du test et qu'aucune date de test
        /// n'est antérieure à une date d'entraînement pour la même province.
        /// </summary>
        private (IDataView Train, IDataView Test) SplitChronologiquementParProvince(IDataView fullData)
        {
            var lignes = MlContext.Data.CreateEnumerable<TRow>(fullData, reuseRowObject: false).ToList();

            var trainRows = new List<TRow>();
            var testRows = new List<TRow>();

            foreach (var groupeProvince in lignes.GroupBy(r => r.Province))
            {
                // L'ordre de chaque groupe préserve l'ordre chronologique du CSV source
                // (garanti par NumericForecastDatasetBuilder, jamais retrié ici).
                var ordonnees = groupeProvince.ToList();
                int indexCoupure = (int)(ordonnees.Count * FractionEntrainement);

                trainRows.AddRange(ordonnees.Take(indexCoupure));
                testRows.AddRange(ordonnees.Skip(indexCoupure));
            }

            return (MlContext.Data.LoadFromEnumerable(trainRows), MlContext.Data.LoadFromEnumerable(testRows));
        }
    }
}
