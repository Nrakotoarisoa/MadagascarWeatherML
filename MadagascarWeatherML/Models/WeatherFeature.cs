using Microsoft.ML.Data;

namespace MadagascarWeatherML.Models
{
    /// <summary>
    /// Représente une ligne du dataset de features enrichies produit par WeatherFeatureEngineer :
    /// une ligne par jour D et par province, avec toutes les variables calendaires, historiques et
    /// géographiques nécessaires. Cette classe sert aussi directement d'entrée (TSrc) aux
    /// PredictionEngine de régression, ses ColumnName correspondant exactement aux noms de
    /// colonnes attendus par les pipelines entraînés (voir RegressionTrainerBase.BuildPipeline).
    ///
    /// Ne contient volontairement aucun label ni horizon : les labels J+1..J+7 sont ajoutés
    /// séparément, directement dans les CSV, par NumericForecastDatasetBuilder — pas ici.
    /// </summary>
    public class WeatherFeature
    {
        // ---------- Identifiants / contexte ----------

        /// <summary>Date de référence D de l'observation (format ISO yyyy-MM-dd).</summary>
        [ColumnName("Date")]
        public string Date { get; set; }

        /// <summary>Province malgache, utilisée en feature catégorielle (One-Hot Encoding).</summary>
        [ColumnName("Province")]
        public string Province { get; set; }

        // ---------- Géographie ----------

        [ColumnName("Latitude")]
        public float Latitude { get; set; }

        [ColumnName("Longitude")]
        public float Longitude { get; set; }

        /// <summary>Altitude approximative du chef-lieu de province (mètres), si disponible.</summary>
        [ColumnName("Altitude")]
        public float Altitude { get; set; }

        // ---------- Features calendaires ----------

        /// <summary>Année civile de la date D.</summary>
        [ColumnName("Annee")]
        public float Annee { get; set; }

        /// <summary>Mois (1-12) à la date D.</summary>
        [ColumnName("Mois")]
        public float Mois { get; set; }

        /// <summary>Trimestre calendaire (1 à 4) de la date D.</summary>
        [ColumnName("Trimestre")]
        public float Trimestre { get; set; }

        /// <summary>Jour de la semaine (0 = dimanche ... 6 = samedi) de la date D.</summary>
        [ColumnName("JourSemaine")]
        public float JourSemaine { get; set; }

        /// <summary>Jour de l'année (1-366) à la date D.</summary>
        [ColumnName("JourAnnee")]
        public float JourAnnee { get; set; }

        /// <summary>Encodage cyclique sinus du jour de l'année (saisonnalité sans discontinuité).</summary>
        [ColumnName("JourAnneeSin")]
        public float JourAnneeSin { get; set; }

        /// <summary>Encodage cyclique cosinus du jour de l'année.</summary>
        [ColumnName("JourAnneeCos")]
        public float JourAnneeCos { get; set; }

        /// <summary>
        /// Saison climatique malgache (feature catégorielle, One-Hot en aval) :
        /// "SaisonChaudeEtPluvieuse" (novembre à avril) ou "SaisonFraicheEtSeche" (mai à octobre).
        /// Ce découpage en 2 saisons est plus pertinent pour Madagascar que les 4 saisons occidentales,
        /// et cohérent avec l'inversion des saisons de l'hémisphère Sud.
        /// </summary>
        [ColumnName("Saison")]
        public string Saison { get; set; }

        // ---------- Lags et moyennes glissantes : température ----------

        [ColumnName("TempLag1")]
        public float TemperatureLag1 { get; set; }

        [ColumnName("TempLag2")]
        public float TemperatureLag2 { get; set; }

        [ColumnName("TempLag3")]
        public float TemperatureLag3 { get; set; }

        [ColumnName("TempLag7")]
        public float TemperatureLag7 { get; set; }

        [ColumnName("TempMoyenneGlissante7j")]
        public float TemperatureRollingMean7 { get; set; }

        [ColumnName("TempEcartTypeGlissant7j")]
        public float TemperatureRollingStd7 { get; set; }

        /// <summary>Moyenne climatologique historique de la température pour ce mois et cette province.</summary>
        [ColumnName("TempClimatologie")]
        public float TemperatureClimatologie { get; set; }

        /// <summary>Écart entre la valeur récente et la climatologie (anomalie court terme).</summary>
        [ColumnName("TempAnomalie")]
        public float TemperatureAnomalie { get; set; }

        // ---------- Lags et moyennes glissantes : précipitation ----------

        [ColumnName("PrecipLag1")]
        public float PrecipitationLag1 { get; set; }

        [ColumnName("PrecipLag3")]
        public float PrecipitationLag3 { get; set; }

        /// <summary>Cumul (somme) des précipitations sur les 7 derniers jours (mm) — plus pertinent qu'une moyenne pour la pluie.</summary>
        [ColumnName("PrecipCumul7j")]
        public float PrecipitationCumul7j { get; set; }

        [ColumnName("PrecipClimatologie")]
        public float PrecipitationClimatologie { get; set; }

        // ---------- Lags et moyennes glissantes : vent ----------

        [ColumnName("VentLag1")]
        public float VentLag1 { get; set; }

        [ColumnName("VentLag3")]
        public float VentLag3 { get; set; }

        [ColumnName("VentMoyenneGlissante7j")]
        public float VentRollingMean7 { get; set; }

        [ColumnName("VentClimatologie")]
        public float VentClimatologie { get; set; }

        // ---------- Autres variables d'état au jour D ----------

        [ColumnName("Humidite")]
        public float Humidite { get; set; }

        [ColumnName("Pression")]
        public float Pression { get; set; }
    }
}
