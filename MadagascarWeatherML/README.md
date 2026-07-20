# Prévision météo Madagascar — ML.NET

Projet ML.NET de prévision météo quotidienne à horizon 7 jours pour les provinces de Madagascar :
Antananarivo, Fianarantsoa, Toamasina, Mahajanga, Toliara et Antsiranana.

Pour une province et une date de début, le système produit J+1 à J+7 avec :

- température moyenne prévue (°C) ;
- précipitations prévues (mm) ;
- vitesse du vent prévue (km/h) ;
- type de météo ;
- score de confiance du classifieur.

Les données d’entrée sont des historiques CSV Open-Meteo, une ligne par jour et par province.

---

## Architecture ML

Le pipeline est volontairement séquentiel pour éviter les fuites de données et conserver une cohérence stricte entre entraînement et production.

1. `OpenMeteoCsvLoader` charge les fichiers CSV par province.
2. `DataCleaner` filtre les lignes invalides, doublons et mesures aberrantes, puis trie par province/date.
3. `WeatherFeatureEngineer` produit les features temporelles, lags, moyennes/cumuls glissants et climatologies sans utiliser le futur.
4. `NumericForecastDatasetBuilder` construit trois datasets de régression multi-horizon : température, précipitations et vent.
5. `TemperatureTrainer`, `PrecipitationTrainer` et `WindTrainer` entraînent 21 modèles de régression (`J1` à `J7` pour chaque cible).
6. `WeatherTypeDatasetBuilder` génère `WeatherTypeTraining.csv` en scorant les 21 modèles de régression déjà entraînés sur l’historique : le classifieur apprend donc sur des valeurs prédites, comme en production.
7. `WeatherTypeTrainer` entraîne 7 classifieurs WeatherType distincts.
8. `WeatherForecastEngine` charge les modèles à la demande, les met en cache et retourne les prévisions chronologiques.

---

## Structure des dossiers

```text
MadagascarWeatherML/
├── DataPreparation/      # chargement, nettoyage, feature engineering
├── Datasets/             # génération des CSV d'entraînement
├── Inference/            # moteur de prédiction 7 jours
├── Models/               # DTO ML.NET et résultats métier + modèles .zip générés
├── Training/             # trainers régression et classification
├── data/                 # CSV Open-Meteo historiques d'entrée
├── datasets/             # CSV d'entraînement générés
├── Program.cs            # orchestration complète du pipeline
└── MadagascarWeatherML.csproj
```

---

## Format des données Open-Meteo

Le loader accepte deux formats :

1. l’ancien format applicatif avec une première ligne d’en-tête `date,latitude,longitude,...` ;
2. l’export quotidien brut Open-Meteo présent dans `data/`, avec des lignes de métadonnées puis une ligne d’en-tête commençant par `time`.

Format applicatif complet :

| # | Colonne | Type | Description |
|---|---|---|---|
| 0 | `date` | string | date ISO `yyyy-MM-dd` |
| 1 | `latitude` | float | latitude du point Open-Meteo |
| 2 | `longitude` | float | longitude du point Open-Meteo |
| 3 | `temperature_moyenne` | float | température moyenne quotidienne |
| 4 | `temperature_min` | float | température minimale |
| 5 | `temperature_max` | float | température maximale |
| 6 | `humidite` | float | humidité relative |
| 7 | `pression` | float | pression atmosphérique |
| 8 | `precipitation` | float | précipitations quotidiennes |
| 9 | `vitesse_vent` | float | vitesse du vent |
| 10 | `direction_vent` | float | direction du vent |
| 11 | `code_meteo` | float | code WMO Open-Meteo |

Pour les exports quotidiens Open-Meteo bruts, le loader mappe notamment `time`, `temperature_2m_min`, `temperature_2m_max`, `weather_code`, `precipitation_sum`, `wind_speed_10m_max` et `wind_direction_10m_dominant`. Si l’export ne contient pas humidité/pression, des valeurs de repli neutres (`75%`, `1013.25 hPa`) sont utilisées afin de conserver l’historique ; pour une qualité supérieure, exportez aussi ces variables depuis Open-Meteo.

La province n’est pas lue depuis le CSV : elle est injectée par le dictionnaire `province -> chemin` fourni au loader.

---

## Features et prévention du data leakage

Les features numériques de régression utilisent uniquement l’état disponible au jour de référence `D` ou avant :

- lags température : `TempLag1`, `TempLag2`, `TempLag3`, `TempLag7` ;
- moyenne et écart-type glissants température sur les 7 jours passés ;
- lags vent : `VentLag1`, `VentLag3` ;
- moyenne glissante vent sur les 7 jours passés ;
- lags précipitations : `PrecipLag1`, `PrecipLag3` ;
- cumul pluie sur les 7 jours passés ;
- climatologies calculées uniquement avec les années strictement antérieures à la date courante ;
- variables calendaires : année, mois, trimestre, jour semaine, jour année, sinus/cosinus annuels, saison malgache ;
- province encodée en one-hot.

Les labels `J+1` à `J+7` sont récupérés par couple exact `(Province, Date cible)`. Si une date future manque, la ligne est rejetée au lieu d’être interpolée.

Le split train/test est chronologique par province : les plus anciennes lignes vont à l’entraînement, les plus récentes au test. Aucun split aléatoire n’est utilisé.

---

## Entraînement des modèles

Exécution complète depuis la racine du dépôt :

```bash
dotnet run --project MadagascarWeatherML/MadagascarWeatherML.csproj
```

Vous pouvez aussi exécuter depuis le dossier projet :

```bash
cd MadagascarWeatherML
dotnet run
```

`Program.cs` résout automatiquement les chemins `data/`, `datasets/` et `Models/` à partir du dossier projet. Cela évite l'erreur `FileNotFoundException: data/antananarivo.csv` lorsque la commande `dotnet run --project ...` est lancée depuis la racine du dépôt.

Séquence logique exécutée par `Program.cs` :

```csharp
var mlContext = new MLContext(seed: 2026);
var loader = new OpenMeteoCsvLoader(mlContext);
var brut = loader.LoadMultipleProvincesAsList(cheminsParProvince);

var propre = new DataCleaner().Clean(brut, out var rapportNettoyage);
var features = new WeatherFeatureEngineer().BuildFeatures(propre);

var generateur = new MLDatasetGenerator();
generateur.GenerateNumericDatasets(propre, features, "datasets");

new TemperatureTrainer().TrainAll("datasets/TemperatureTraining.csv", "Models");
new PrecipitationTrainer().TrainAll("datasets/PrecipitationTraining.csv", "Models");
new WindTrainer().TrainAll("datasets/WindTraining.csv", "Models");

generateur.GenerateWeatherTypeDataset(propre, features, "Models", "datasets", mlContext);
new WeatherTypeTrainer().TrainAll("datasets/WeatherTypeTraining.csv", "Models");
```

Les modèles de régression testent LightGbm, FastTree et FastForest. Le meilleur modèle est sélectionné par RMSE minimal, puis réentraîné sur tout le dataset disponible de l’horizon.

Les modèles WeatherType testent LightGbmMulticlass, SdcaMaximumEntropy et FastTree via OneVersusAll. Le meilleur modèle est sélectionné automatiquement selon :

1. `MacroAccuracy` la plus élevée ;
2. `F1 Macro` le plus élevé en cas d’égalité ;
3. `LogLoss` la plus faible en cas de nouvelle égalité.

---

## Fichiers générés

Après entraînement, `Models/` contient 28 fichiers `.zip` :

```text
Temperature_J1.zip   ... Temperature_J7.zip
Precipitation_J1.zip ... Precipitation_J7.zip
Wind_J1.zip          ... Wind_J7.zip
WeatherType_J1.zip   ... WeatherType_J7.zip
```

Les datasets générés dans `datasets/` sont :

- `TemperatureTraining.csv` ;
- `PrecipitationTraining.csv` ;
- `WindTraining.csv` ;
- `WeatherTypeTraining.csv`.

`WeatherTypeTraining.csv` contient la date cible, l’horizon, les sorties prévues par les modèles de régression, humidité/pression persistées depuis `D`, la province et les variables temporelles de la date cible.

---

## Exécution d’une prédiction 7 jours

```csharp
var moteur = new WeatherForecastEngine("Models", propre);

List<WeatherForecastResult> previsions = moteur.Predict(
    province: "Antananarivo",
    startDate: "2026-10-01",
    days: 7);
```

Le moteur utilise `startDate - 1` comme date de référence historique `D`, puis applique les modèles `J1` à `J7`. Les résultats sont retournés dans l’ordre chronologique.

Exemple de sortie :

```json
{
  "date": "2026-10-01",
  "province": "Antananarivo",
  "temperature": 25.4,
  "precipitation": 8.2,
  "windSpeed": 18,
  "weatherType": "Pluvieux",
  "confidence": 0.87
}
```

---

## Métriques d’évaluation

### Régression

Chaque horizon de température, précipitations et vent expose :

- RMSE ;
- MAE ;
- R² ;
- comparaison des algorithmes candidats.

### Classification WeatherType

Chaque horizon expose :

- MacroAccuracy ;
- MicroAccuracy ;
- F1 Macro calculé depuis la matrice de confusion ;
- LogLoss ;
- matrice de confusion formatée.

---

## Réentraînement avec de nouvelles données

1. Ajouter ou remplacer les CSV Open-Meteo dans `data/`.
2. Relancer tout le pipeline dans l’ordre complet.
3. Ne pas réutiliser un ancien `WeatherTypeTraining.csv` après réentraînement des régressions : il dépend des sorties des modèles de régression courants.
4. Remplacer les `.zip` en production uniquement après validation des métriques.
5. Conserver un rapport d’entraînement par date de réentraînement pour tracer les métriques et les algorithmes retenus.

---

## Maintenance

- Vérifier régulièrement que les CSV restent chronologiques et sans trous importants.
- Surveiller les métriques par province et par horizon : J+7 doit naturellement être moins précis que J+1.
- Réentraîner si la distribution des données change fortement (nouvel historique, changement de source Open-Meteo, ajout de province).
- Pour une API ASP.NET Core, préférer `PredictionEnginePool` plutôt qu’un singleton partagé, car `PredictionEngine` n’est pas thread-safe.
- Garder les packages ML.NET cohérents entre eux (`Microsoft.ML`, `Microsoft.ML.FastTree`, `Microsoft.ML.LightGbm`).

---

## Rapport d’audit final

### Problèmes détectés

- Le loader ne prenait pas correctement en charge les exports Open-Meteo quotidiens bruts présents dans `data/`.
- Le classifieur WeatherType ne recevait pas explicitement les variables temporelles complètes demandées.
- `WeatherTypeTraining.csv` ne contenait pas la date cible, ce qui rendait le contrôle chronologique moins robuste.
- La sélection automatique du meilleur classifieur utilisait F1 Macro comme critère principal au lieu de MacroAccuracy puis F1 Macro puis LogLoss.
- Le programme de démonstration affichait le meilleur classifieur selon l’ancien critère.
- La documentation ne couvrait pas assez précisément les métriques, la maintenance et la structure ML complète.

### Corrections effectuées

- Loader rendu compatible avec les exports Open-Meteo quotidiens bruts et l’ancien format applicatif.
- Ajout de la date cible et des variables calendaires dans `WeatherTypeTraining.csv`.
- Ajout des mêmes variables calendaires dans l’entrée de scoring WeatherType du moteur d’inférence.
- Split WeatherType trié explicitement par date à l’intérieur de chaque province.
- Sélection du classifieur alignée sur les critères demandés : MacroAccuracy, F1 Macro, LogLoss.
- Conservation de la matrice de confusion formatée dans le résultat d’évaluation classification.
- README complété pour décrire l’architecture, les datasets, l’entraînement, les `.zip`, la prédiction, les métriques et la maintenance.

### Optimisations appliquées

- Pas de changement architectural inutile.
- Réutilisation des caches de `PredictionEngine` déjà présents.
- Alignement entraînement/production du classifieur sans duplication de pipeline.
- Contrôle chronologique WeatherType rendu explicite via la date cible.

### Points restant à améliorer

- Installer le SDK .NET dans l’environnement d’exécution CI pour compiler et exécuter le pipeline automatiquement.
- Ajouter des tests unitaires sur le tri chronologique, les lags et la génération des horizons.
- Ajouter un export persistant des rapports de métriques en JSON/CSV.
- En production web, remplacer les `PredictionEngine` cachés par `PredictionEnginePool`.

## Dépannage

### `ArgumentOutOfRangeException: Must be less than lowerBound (Parameter 'upperBound')`

Cette erreur venait du filtrage des lignes WeatherType par horizon avec une plage dégénérée `lowerBound == upperBound` dans `FilterRowsByColumn`. ML.NET exige une borne supérieure strictement supérieure à la borne inférieure. Le filtrage utilise désormais une plage autour de l'entier d'horizon (`h - 0.5` à `h + 0.5`) afin de sélectionner uniquement les lignes `Horizon = h` sans déclencher l'exception.

### `Could not find input column 'WeatherType'`

Cette erreur peut survenir en prédiction si le modèle WeatherType sauvegardé contient encore le transform de label `MapValueToKey("Label", "WeatherType")`. Même si le label n'est pas utilisé pour scorer, ML.NET exige que la colonne source existe quand il construit le `PredictionEngine`. L'entrée de scoring fournit donc une valeur factice `WeatherType = "Autre"`; elle ne sert pas à la prédiction mais satisfait le schéma attendu par la chaîne de transforms.

## API ASP.NET Core d'inférence

Cette API expose uniquement les modèles ML.NET déjà entraînés dans `Models/` ; elle ne déclenche aucun entraînement et ne modifie jamais les fichiers `.zip`.

### Analyse et choix d'architecture

- Les modèles sont chargés une seule fois au démarrage par `ModelRegistry`, puis conservés en mémoire pour éviter les accès disque à chaque requête.
- Les services de prédiction spécialisés (`TemperaturePredictionService`, `PrecipitationPredictionService`, `WindPredictionService`, `WeatherTypePredictionService`) isolent chaque famille de modèles, tandis que `WeatherForecastService` orchestre le scénario métier complet.
- `PredictionEngine` n'étant pas thread-safe, l'API utilise un moteur par famille/horizon protégé par verrou. Cette stratégie est simple et sûre ; elle pourra être remplacée par `PredictionEnginePool` si un débit plus élevé est requis.
- L'architecture DI et les extensions `AddWeatherApi` / `AddWeatherSwagger` préparent l'ajout ultérieur de JWT, rate limiting et versionnement sans modifier les contrôleurs.

### Endpoints

- `GET /api/weather/forecast?province=Antananarivo&startDate=2026-10-01&days=7`
- `GET /api/weather/day?province=Antananarivo&date=2026-10-01`
- `GET /api/provinces`
- `GET /api/models/status`

La documentation interactive est disponible via Swagger à l'adresse `/swagger`.
