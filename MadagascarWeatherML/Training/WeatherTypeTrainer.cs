using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MadagascarWeatherML.Training
{
    /// <summary>
    /// Schéma de WeatherTypeTraining.csv (produit par WeatherTypeDatasetBuilder, après entraînement
    /// des modèles de régression). TemperatureFuture/PrecipitationFuture/VentFutur sont les sorties
    /// RÉELLES des modèles de régression scorés sur l'historique — pas des valeurs observées —,
    /// ce qui correspond exactement à ce que le classifieur recevra en production.
    /// Horizon (1 à 7) permet de sélectionner, à l'entraînement, le sous-ensemble propre à chaque
    /// modèle WeatherType_Jx (le bruit des prédictions de régression augmente avec l'horizon).
    /// </summary>
    public class WeatherTypeTrainingRow
    {
        [LoadColumn(0)] [ColumnName("Province")] public string Province { get; set; }
        [LoadColumn(1)] [ColumnName("Date")] public string Date { get; set; }
        [LoadColumn(2)] [ColumnName("Horizon")] public float Horizon { get; set; }
        [LoadColumn(3)] [ColumnName("TemperatureFuture")] public float TemperatureFuture { get; set; }
        [LoadColumn(4)] [ColumnName("PrecipitationFuture")] public float PrecipitationFuture { get; set; }
        [LoadColumn(5)] [ColumnName("VentFutur")] public float VentFutur { get; set; }
        [LoadColumn(6)] [ColumnName("Humidite")] public float Humidite { get; set; }
        [LoadColumn(7)] [ColumnName("Pression")] public float Pression { get; set; }
        [LoadColumn(8)] [ColumnName("Annee")] public float Annee { get; set; }
        [LoadColumn(9)] [ColumnName("Mois")] public float Mois { get; set; }
        [LoadColumn(10)] [ColumnName("Trimestre")] public float Trimestre { get; set; }
        [LoadColumn(11)] [ColumnName("JourSemaine")] public float JourSemaine { get; set; }
        [LoadColumn(12)] [ColumnName("JourAnnee")] public float JourAnnee { get; set; }
        [LoadColumn(13)] [ColumnName("JourAnneeSin")] public float JourAnneeSin { get; set; }
        [LoadColumn(14)] [ColumnName("JourAnneeCos")] public float JourAnneeCos { get; set; }
        [LoadColumn(15)] [ColumnName("Saison")] public string Saison { get; set; }
        [LoadColumn(16)] [ColumnName("WeatherType")] public string WeatherType { get; set; }
    }

    /// <summary>Résultat d'évaluation d'un algorithme candidat de classification, pour un horizon donné.</summary>
    public class ClassificationEvaluationResult
    {
        public int Horizon { get; set; }
        public string Algorithme { get; set; }
        public double MacroAccuracy { get; set; }
        public double MicroAccuracy { get; set; }
        public double F1Macro { get; set; }
        public double LogLoss { get; set; }
        public string ConfusionMatrix { get; set; }
    }

    /// <summary>
    /// Entraîne les 7 modèles de classification du type de météo (WeatherType_J1.zip à
    /// WeatherType_J7.zip) à partir de WeatherTypeTraining.csv. Pour chaque horizon, filtre les
    /// lignes correspondantes (colonne Horizon), teste LightGbm multiclasse, SdcaMaximumEntropy et
    /// FastTree multiclasse (via OneVersusAll, ML.NET ne proposant pas de FastTree multiclasse
    /// natif), puis sélectionne automatiquement le meilleur sur le F1 Macro.
    ///
    /// Correction par rapport à une version précédente : les 7 modèles sont désormais réellement
    /// distincts (un entraînement par horizon), et non un unique modèle dupliqué 7 fois. Ce
    /// changement est nécessaire car TemperatureFuture/PrecipitationFuture/VentFutur sont désormais
    /// les sorties réelles des modèles de régression : leur distribution de bruit dépend de
    /// l'horizon (J+7 nettement plus incertain que J+1), donc le classifieur optimal peut différer
    /// d'un horizon à l'autre.
    ///
    /// Note de cohérence projet : WeatherTypeTraining.csv est construit avec 6 catégories
    /// (Ensoleille, Nuageux, Pluvieux, Orageux, Brouillard, Autre — voir enum WeatherType), la
    /// 6e servant de repli pour les codes météo WMO non couverts par Sunny/Cloudy/Rain/Storm/Fog.
    /// Mapping conceptuel : Sunny=Ensoleille, Cloudy=Nuageux, Rain=Pluvieux, Storm=Orageux,
    /// Fog=Brouillard, Autre=repli. Le pipeline reste entièrement piloté par les valeurs réellement
    /// présentes dans le CSV (MapValueToKey), donc aucun changement de code n'est requis si les
    /// libellés évoluent.
    /// </summary>
    public class WeatherTypeTrainer
    {
        private const int HorizonMax = 7;
        private const double FractionEntrainement = 0.8;

        private readonly MLContext _mlContext;

        public WeatherTypeTrainer(int seed = 2026)
        {
            _mlContext = new MLContext(seed: seed);
        }

        /// <summary>
        /// Entraîne les 7 horizons et écrit WeatherType_J1.zip ... WeatherType_J7.zip dans
        /// modelsOutputDirectory. Doit être appelée après la génération de WeatherTypeTraining.csv
        /// (elle-même postérieure à l'entraînement des modèles de régression — voir README.md).
        /// Peut être rappelée à tout moment avec un csvPath mis à jour pour réentraîner sur de
        /// nouvelles données (aucun état conservé entre deux appels).
        /// </summary>
        public List<ClassificationEvaluationResult> TrainAll(string csvPath, string modelsOutputDirectory)
        {
            if (!File.Exists(csvPath))
                throw new FileNotFoundException("Dataset d'entraînement introuvable.", csvPath);

            IDataView fullData = _mlContext.Data.LoadFromTextFile<WeatherTypeTrainingRow>(
                csvPath, hasHeader: true, separatorChar: ',');

            Directory.CreateDirectory(modelsOutputDirectory);

            var rapportComplet = new List<ClassificationEvaluationResult>();

            for (int h = 1; h <= HorizonMax; h++)
            {
                // Isole les lignes de cet horizon. FilterRowsByColumn exige upperBound > lowerBound
                // (borne supérieure exclusive côté ML.NET), donc on encadre l'entier h plutôt que
                // d'utiliser une plage dégénérée [h, h] qui déclenche ArgumentOutOfRangeException.
                IDataView dataHorizon = _mlContext.Data.FilterRowsByColumn(
                    fullData, "Horizon", lowerBound: h - 0.5, upperBound: h + 0.5);

                var resultatHorizon = TrainHorizon(dataHorizon, h, modelsOutputDirectory);
                rapportComplet.AddRange(resultatHorizon);
            }

            return rapportComplet;
        }

        private List<ClassificationEvaluationResult> TrainHorizon(IDataView dataHorizon, int horizon, string modelsOutputDirectory)
        {
            var (trainData, testData) = SplitChronologiquementParProvince(dataHorizon);

            var candidats = new (string Nom, IEstimator<ITransformer> Pipeline)[]
            {
                ("LightGbmMulticlass", BuildPipeline(
                    _mlContext.MulticlassClassification.Trainers.LightGbm(
                        labelColumnName: "Label", featureColumnName: "Features"))),

                ("SdcaMaximumEntropy", BuildPipeline(
                    _mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                        labelColumnName: "Label", featureColumnName: "Features"))),

                ("FastTreeMulticlass_OneVersusAll", BuildPipeline(
                    _mlContext.MulticlassClassification.Trainers.OneVersusAll(
                        _mlContext.BinaryClassification.Trainers.FastTree(
                            labelColumnName: "Label", featureColumnName: "Features"),
                        labelColumnName: "Label")))
            };

            var evaluations = new List<(string Nom, MulticlassClassificationMetrics Metrics)>();

            foreach (var (nom, pipeline) in candidats)
            {
                ITransformer modele = pipeline.Fit(trainData);
                IDataView predictions = modele.Transform(testData);

                MulticlassClassificationMetrics metrics = _mlContext.MulticlassClassification.Evaluate(
                    predictions,
                    labelColumnName: "Label",
                    scoreColumnName: "Score",
                    predictedLabelColumnName: "PredictedLabel");

                evaluations.Add((nom, metrics));
            }

            var rapport = evaluations.Select(e => new ClassificationEvaluationResult
            {
                Horizon = horizon,
                Algorithme = e.Nom,
                MacroAccuracy = e.Metrics.MacroAccuracy,
                MicroAccuracy = e.Metrics.MicroAccuracy,
                F1Macro = CalculerF1Macro(e.Metrics),
                LogLoss = e.Metrics.LogLoss,
                ConfusionMatrix = e.Metrics.ConfusionMatrix.GetFormattedConfusionTable()
            }).ToList();

            // Sélection automatique conforme au cahier des charges : MacroAccuracy en critère
            // principal, F1 Macro pour départager les égalités, puis LogLoss minimal.
            var meilleur = rapport
                .OrderByDescending(r => r.MacroAccuracy)
                .ThenByDescending(r => r.F1Macro)
                .ThenBy(r => r.LogLoss)
                .First();
            var pipelineRetenu = candidats.First(c => c.Nom == meilleur.Algorithme).Pipeline;

            // Réentraînement final sur l'intégralité des données disponibles pour cet horizon.
            ITransformer modeleFinal = pipelineRetenu.Fit(dataHorizon);

            string cheminModele = Path.Combine(modelsOutputDirectory, $"WeatherType_J{horizon}.zip");
            _mlContext.Model.Save(modeleFinal, dataHorizon.Schema, cheminModele);

            return rapport;
        }

        /// <summary>
        /// Pipeline complet : encodage catégoriel (Province, Saison), concaténation des features,
        /// conversion du label texte en clé (MapValueToKey, requis par les trainers multiclasse),
        /// puis reconversion de la prédiction en libellé texte lisible (MapKeyToValue) — le modèle
        /// sauvegardé produit donc directement une chaîne ("Pluvieux", etc.) et non une clé numérique.
        /// Horizon n'est volontairement pas inclus dans Features : il sert uniquement à isoler le
        /// sous-ensemble d'entraînement propre à ce modèle, pas de feature au sein d'un même modèle.
        /// </summary>
        private IEstimator<ITransformer> BuildPipeline(IEstimator<ITransformer> trainer)
        {
            return _mlContext.Transforms.Conversion.MapValueToKey("Label", "WeatherType")
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("ProvinceEncoded", "Province"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("SaisonEncoded", "Saison"))
                .Append(_mlContext.Transforms.Concatenate("Features",
                    "TemperatureFuture", "PrecipitationFuture", "VentFutur",
                    "Humidite", "Pression",
                    "Annee", "Mois", "Trimestre", "JourSemaine", "JourAnnee",
                    "JourAnneeSin", "JourAnneeCos",
                    "ProvinceEncoded", "SaisonEncoded"))
                .Append(trainer)
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
        }

        /// <summary>
        /// ML.NET n'expose pas directement un "F1 Macro" dans MulticlassClassificationMetrics :
        /// on le calcule à partir de la précision et du rappel par classe de la matrice de
        /// confusion (F1_classe = 2·P·R / (P+R), puis moyenne non pondérée sur toutes les classes).
        /// </summary>
        private static double CalculerF1Macro(MulticlassClassificationMetrics metrics)
        {
            var precisions = metrics.ConfusionMatrix.PerClassPrecision;
            var rappels = metrics.ConfusionMatrix.PerClassRecall;

            var f1ParClasse = new List<double>();

            for (int i = 0; i < precisions.Count; i++)
            {
                double p = precisions[i];
                double r = rappels[i];
                f1ParClasse.Add(p + r == 0 ? 0 : 2 * p * r / (p + r));
            }

            return f1ParClasse.Count == 0 ? 0 : f1ParClasse.Average();
        }

        /// <summary>
        /// Split chronologique strict par province (mêmes garanties que pour les entraîneurs de
        /// régression) : 80% les plus anciens en entraînement, 20% les plus récents en test,
        /// indépendamment pour chaque province — jamais de mélange futur/passé.
        /// </summary>
        private (IDataView Train, IDataView Test) SplitChronologiquementParProvince(IDataView dataHorizon)
        {
            var lignes = _mlContext.Data.CreateEnumerable<WeatherTypeTrainingRow>(dataHorizon, reuseRowObject: false).ToList();

            var trainRows = new List<WeatherTypeTrainingRow>();
            var testRows = new List<WeatherTypeTrainingRow>();

            foreach (var groupeProvince in lignes.GroupBy(r => r.Province))
            {
                var ordonnees = groupeProvince
                    .OrderBy(r => r.Date, StringComparer.Ordinal)
                    .ToList();
                int indexCoupure = (int)(ordonnees.Count * FractionEntrainement);

                trainRows.AddRange(ordonnees.Take(indexCoupure));
                testRows.AddRange(ordonnees.Skip(indexCoupure));
            }

            return (_mlContext.Data.LoadFromEnumerable(trainRows), _mlContext.Data.LoadFromEnumerable(testRows));
        }
    }
}
