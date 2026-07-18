# Prévision météo Madagascar — Projet ML.NET

Système de prévision météo à 7 jours pour les 6 provinces de Madagascar (Antananarivo,
Fianarantsoa, Toamasina, Mahajanga, Toliara, Antsiranana), basé sur ML.NET et des historiques
Open-Meteo. Ce document couvre uniquement la partie Machine Learning (aucun backend ASP.NET Core
n'est inclus dans ce dépôt).

## Sommaire

1. [Format des données Open-Meteo](#1-format-des-données-open-meteo)
2. [Ajouter une nouvelle province](#2-ajouter-une-nouvelle-province)
3. [Ordre d'exécution du pipeline](#3-ordre-dexécution-du-pipeline)
4. [Entraînement des modèles](#4-entraînement-des-modèles)
5. [Génération des fichiers .zip](#5-génération-des-fichiers-zip)
6. [Prédiction sur 7 jours](#6-prédiction-sur-7-jours)
7. [Réentraînement avec de nouvelles données](#7-réentraînement-avec-de-nouvelles-données)
8. [Corrections apportées lors de l'audit](#8-corrections-apportées-lors-de-laudit)

---

## 1. Format des données Open-Meteo

Chaque province dispose de son propre fichier CSV historique (un export Open-Meteo par
coordonnées lat/lon du chef-lieu de province). `OpenMeteoCsvLoader` attend, **avec en-tête**, les
colonnes suivantes, dans cet ordre exact :

| # | Colonne         | Type   | Exemple      |
|---|------------------|--------|--------------|
| 0 | date             | string | 2024-03-15   |
| 1 | latitude         | float  | -18.8792     |
| 2 | longitude        | float  | 47.5079      |
| 3 | temperature_moyenne | float | 24.3      |
| 4 | temperature_min  | float  | 18.1         |
| 5 | temperature_max  | float  | 29.6         |
| 6 | humidite         | float  | 78.0 (%)     |
| 7 | pression         | float  | 1012.5 (hPa) |
| 8 | precipitation    | float  | 12.4 (mm)    |
| 9 | vitesse_vent     | float  | 14.2 (km/h)  |
| 10| direction_vent   | float  | 210.0 (°)    |
| 11| code_meteo       | float  | 61 (code WMO)|

Le champ **Province n'est volontairement pas dans le CSV** (les exports Open-Meteo bruts n'ont pas
cette notion) : il est injecté par `OpenMeteoCsvLoader` à partir du dictionnaire fourni en entrée
(nom de province → chemin du fichier). `code_meteo` est le code WMO standard (0 = ciel clair,
61/63/65 = pluie, 95/96/99 = orage, 45/48 = brouillard, etc.) — voir le mapping complet dans
`WeatherTypeDatasetBuilder.MapCodeMeteoToWeatherType`.

Une ligne manquante, mal formée, ou avec une valeur non numérique est marquée `NaN` par le loader
puis éliminée par `DataCleaner` (jamais silencieusement corrigée).

## 2. Ajouter une nouvelle province

1. **Obtenir l'historique Open-Meteo** de la nouvelle province (même format que ci-dessus), sur la
   plus longue période disponible.
2. **Ajouter une entrée** dans le dictionnaire passé à `OpenMeteoCsvLoader.LoadMultipleProvincesAsList`,
   par ex. `{ "Itasy", "data/itasy.csv" }`.
3. **Ajouter le nom de la province** à la liste `ProvincesValides` dans `WeatherForecastEngine.cs`
   (sinon `Predict` la rejettera).
4. *(Optionnel)* Ajouter son altitude dans le dictionnaire `provinceAltitudes` transmis à
   `WeatherFeatureEngineer.BuildFeatures`.
5. **Relancer tout le pipeline** (section 3) : `Province` est une feature catégorielle apprise par
   One-Hot Encoding, donc les 28 modèles doivent être réentraînés pour "connaître" la nouvelle
   province — il n'existe pas de mécanisme d'ajout incrémental d'une catégorie à un modèle déjà
   entraîné en ML.NET.
6. Vérifier qu'il y a **au moins ~30 jours d'historique continu** pour la nouvelle province avant
   de lancer une prédiction (7 jours minimum pour les lags, plus une marge pour la climatologie).

## 3. Ordre d'exécution du pipeline

Point important corrigé lors de l'audit : **`WeatherTypeTraining.csv` dépend désormais des 21
modèles de régression déjà entraînés** (voir section 8). L'ordre d'exécution n'est donc plus
librement parallélisable et doit respecter :

```
1. OpenMeteoCsvLoader.LoadMultipleProvincesAsList(...)         -> List<WeatherRawData> (brut)
2. DataCleaner.Clean(...)                                       -> List<WeatherRawData> (nettoyé, trié)
3. WeatherFeatureEngineer.BuildFeatures(...)                     -> List<WeatherFeature>
4. MLDatasetGenerator.GenerateNumericDatasets(...)               -> Temperature/Precipitation/WindTraining.csv
5. TemperatureTrainer.TrainAll(...)                              -> Models/Temperature_J1..J7.zip
   PrecipitationTrainer.TrainAll(...)                            -> Models/Precipitation_J1..J7.zip
   WindTrainer.TrainAll(...)                                     -> Models/Wind_J1..J7.zip
6. MLDatasetGenerator.GenerateWeatherTypeDataset(...)            -> WeatherTypeTraining.csv
   (nécessite les 21 modèles de l'étape 5 : ils sont scorés sur l'historique)
7. WeatherTypeTrainer.TrainAll(...)                              -> Models/WeatherType_J1..J7.zip
8. new WeatherForecastEngine(modelsDirectory, cleanedData).Predict(...)
```

## 4. Entraînement des modèles

```csharp
var mlContext = new MLContext(seed: 2026);

var loader = new OpenMeteoCsvLoader(mlContext);
var brut = loader.LoadMultipleProvincesAsList(new Dictionary<string, string>
{
    ["Antananarivo"]  = "data/antananarivo.csv",
    ["Fianarantsoa"] = "data/fianarantsoa.csv",
    ["Toamasina"]   = "data/toamasina.csv",
    ["Mahajanga"]   = "data/mahajanga.csv",
    ["Toliara"]     = "data/toliara.csv",
    ["Antsiranana"] = "data/antsiranana.csv",
});

var cleaner = new DataCleaner();
var propre = cleaner.Clean(brut, out var rapportNettoyage);

var features = new WeatherFeatureEngineer().BuildFeatures(propre);

var generateur = new MLDatasetGenerator();
generateur.GenerateNumericDatasets(propre, features, "datasets/");

// Étape 5 : régressions (chacune teste LightGbm / FastTree / FastForest et choisit automatiquement)
new TemperatureTrainer().TrainAll("datasets/TemperatureTraining.csv", "Models/");
new PrecipitationTrainer().TrainAll("datasets/PrecipitationTraining.csv", "Models/");
new WindTrainer().TrainAll("datasets/WindTraining.csv", "Models/");

// Étape 6 : dataset du classifieur, généré à partir des modèles de régression ci-dessus
generateur.GenerateWeatherTypeDataset(propre, features, regressionModelsDirectory: "Models/", outputDirectory: "datasets/");

// Étape 7 : classification (teste LightGbm / SdcaMaximumEntropy / FastTree OneVersusAll)
new WeatherTypeTrainer().TrainAll("datasets/WeatherTypeTraining.csv", "Models/");
```

Chaque `TrainAll` retourne la liste des métriques comparées par algorithme et par horizon
(`HorizonTrainingResult` pour les régressions, `ClassificationEvaluationResult` pour la
classification) — utile à consigner dans un rapport académique.

## 5. Génération des fichiers .zip

À l'issue de l'étape 4, `Models/` contient 28 fichiers :

```
Models/
├── Temperature_J1.zip   ... Temperature_J7.zip
├── Precipitation_J1.zip ... Precipitation_J7.zip
├── Wind_J1.zip          ... Wind_J7.zip
└── WeatherType_J1.zip   ... WeatherType_J7.zip
```

Les 21 modèles de régression sont réellement distincts (un par horizon, chacun réentraîné sur
l'intégralité des données après sélection automatique du meilleur algorithme). Les 7 modèles de
classification sont **également réellement distincts entre eux** depuis la correction de l'audit
(section 8) — contrairement à une version précédente qui dupliquait un modèle unique.

## 6. Prédiction sur 7 jours

```csharp
var propre = /* DataCleaner.Clean(...), déjà fait une fois */;

var moteur = new WeatherForecastEngine(modelsDirectory: "Models/", cleanedHistoricalData: propre);

List<WeatherForecastResult> previsions = moteur.Predict(
    province: "Antananarivo",
    startDate: "2026-10-01",
    days: 7);
```

Retourne une liste de `WeatherForecastResult` (une entrée par jour) :

```json
[
  {
    "date": "2026-10-01",
    "province": "Antananarivo",
    "temperature": 25.4,
    "precipitation": 8.2,
    "windSpeed": 18,
    "weatherType": "Pluvieux",
    "confidence": 0.87
  }
]
```

`WeatherForecastEngine` doit disposer d'au moins 7 jours d'historique continu avant `startDate`
pour la province demandée (sinon `InvalidOperationException`). Une instance peut servir plusieurs
appels `Predict` (elle met en cache les modèles chargés) ; voir la note de thread-safety dans le
code pour un usage en API.

## 7. Réentraînement avec de nouvelles données

Aucune classe de ce projet ne conserve d'état entre deux exécutions : réentraîner avec de
nouvelles données consiste simplement à **rejouer le pipeline de la section 3 avec un historique
mis à jour**, dans le même ordre :

1. Ajouter les nouvelles lignes aux CSV Open-Meteo (ou fournir des fichiers mis à jour).
2. Relancer les étapes 1 à 3 (chargement, nettoyage, feature engineering).
3. Relancer l'étape 4 (régénère les 3 CSV de régression avec l'historique complet).
4. Relancer l'étape 5 : `TrainAll` **écrase** les `.zip` existants avec les nouveaux modèles.
5. Relancer l'étape 6 (le CSV du classifieur doit être régénéré avec les *nouvelles* sorties de
   régression — ne pas réutiliser un ancien `WeatherTypeTraining.csv`).
6. Relancer l'étape 7.

Il n'y a pas de réentraînement incrémental (fine-tuning) : chaque appel à `TrainAll` réentraîne
entièrement depuis le CSV fourni. C'est volontaire — les trainers ML.NET utilisés (LightGbm,
FastTree, FastForest, SdcaMaximumEntropy) ne supportent pas l'apprentissage incrémental, et un
réentraînement complet reste peu coûteux à l'échelle de ce projet (6 provinces, prévisions
journalières).

## 8. Corrections apportées lors de l'audit

| # | Problème identifié | Correction |
|---|---|---|
| 1 | Fichier fantôme `TrainingDatasetRows.cs` dupliquant des classes déjà définies (namespace `Models` au lieu de `Training`), non référencé nulle part. | Supprimé. |
| 2 | **Fuite de distribution majeure** : `WeatherTypeTraining.csv` utilisait les valeurs météo réellement observées comme "valeurs futures", alors qu'en production le classifieur reçoit les sorties (imparfaites) des modèles de régression. Le classifieur apprenait donc sur des données plus propres que celles qu'il reçoit réellement en production. | `WeatherTypeDatasetBuilder` charge désormais les 21 modèles de régression déjà entraînés et les score sur l'historique pour produire les features d'entraînement — exactement ce que fait `WeatherForecastEngine` en production. |
| 3 | Conséquence directe du point 2 : le bruit de prédiction dépend de l'horizon (J+7 plus incertain que J+1), donc un classifieur unique dupliqué 7 fois n'était plus justifié. | `WeatherTypeTraining.csv` inclut désormais une colonne `Horizon` ; `WeatherTypeTrainer` entraîne 7 modèles réellement distincts (filtrage via `FilterRowsByColumn`). |
| 4 | Incohérence Humidité/Pression : l'ancienne version utilisait les valeurs réelles du jour cible D+h, jamais disponibles en production. | Aligné sur le jour de référence D (persistance), identique à `WeatherForecastEngine`. |
| 5 | `WeatherFeature.cs` contenait des champs `Horizon`/`Label*` vestiges de la toute première architecture (modèle unique multi-horizon), jamais renseignés ni lus par le pipeline réellement implémenté. | Supprimés. |
| 6 | `WeatherTypePredictionResult` (Models) déclarait `PredictedLabel` comme clé `uint` (`KeyType`), alors que le pipeline sauvegardé reconvertit déjà la clé en texte (`MapKeyToValue`). | Corrigé en `string`, aligné sur `WeatherTypeScoringOutput` réellement utilisé par le moteur d'inférence. |
| 7 | Incohérence de nommage de province dans les commentaires (`Antananarivo` dans `WeatherRawData.cs`/`OpenMeteoCsvLoader.cs` vs `Antananarivo` dans `WeatherForecastEngine.cs`, qui fait foi car c'est la validation réellement exécutée). | Harmonisé sur `Antananarivo` partout. |
| 8 | `MLDatasetGenerator.GenerateAllDatasets` masquait la nouvelle dépendance d'ordre (point 2) en générant les 4 CSV en un seul appel. | Scindé en `GenerateNumericDatasets` (aucune dépendance) et `GenerateWeatherTypeDataset` (nécessite les modèles de régression), avec la séquence documentée en section 3. |

Points vérifiés sans anomalie détectée :

- **Compilation** : namespaces cohérents (`Models`, `DataPreparation`, `Datasets`, `Training`,
  `Inference`), pas de référence à un type ou une propriété inexistante après corrections.
- **Cohérence features entraînement/prédiction** : les `ColumnName` de `WeatherFeature` (Models)
  et `CommonWeatherFeatureRow` (Training) correspondent exactement (27 colonnes), ce qui permet
  d'utiliser `WeatherFeature` directement comme entrée des `PredictionEngine` de régression sans
  classe dupliquée.
- **Absence de fuite temporelle** : recherche des labels futurs par date calendaire exacte
  (jamais par décalage d'index), climatologie calculée uniquement sur les années strictement
  antérieures, split train/test chronologique par province (jamais aléatoire).
- **Performance** : les `PredictionEngine` sont mis en cache par horizon (pas de rechargement de
  modèle à chaque prédiction) ; volumétrie (6 provinces, séries journalières) largement dans les
  capacités de ML.NET sans optimisation particulière. Point d'attention documenté : plusieurs
  classes créent chacune leur propre `MLContext` — sans impact fonctionnel à cette échelle, mais
  un `MLContext` partagé (paramètre optionnel déjà présent sur `WeatherTypeDatasetBuilder` et
  `MLDatasetGenerator.GenerateWeatherTypeDataset`) est préférable si le projet grandit.
