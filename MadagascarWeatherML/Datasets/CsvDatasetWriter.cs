using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace MadagascarWeatherML.Datasets
{
    /// <summary>
    /// Écriture CSV bas niveau, partagée par tous les générateurs de datasets d'entraînement.
    /// N'effectue aucun tri ni filtrage : les lignes sont écrites dans l'ordre où elles sont
    /// fournies, afin de préserver l'ordre chronologique strict établi en amont du pipeline.
    /// </summary>
    public static class CsvDatasetWriter
    {
        public static void Write(string outputPath, IReadOnlyList<string> header, IEnumerable<string[]> rows)
        {
            string directory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);

            using var writer = new StreamWriter(outputPath, append: false, encoding: Encoding.UTF8);

            writer.WriteLine(string.Join(",", header.Select(Escape)));

            foreach (var row in rows)
                writer.WriteLine(string.Join(",", row.Select(Escape)));
        }

        private static string Escape(string value)
        {
            if (string.IsNullOrEmpty(value))
                return string.Empty;

            if (value.IndexOfAny(new[] { ',', '"', '\n', '\r' }) >= 0)
                return "\"" + value.Replace("\"", "\"\"") + "\"";

            return value;
        }
    }
}
