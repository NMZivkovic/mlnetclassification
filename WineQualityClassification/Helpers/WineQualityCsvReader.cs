using System.Collections.Generic;
using System.IO;
using System.Linq;
using WineQualityClassification.WineQualityData;

namespace WineQualityClassification.Helpers
{
    public class WineQualityCsvReader
    {
        public IEnumerable<WineQualitySample> GetWineQualitySamplesFromCsv(string dataLocation)
        {
            return File.ReadAllLines(dataLocation)
               .Skip(1)
               .Select(x => x.Split(';'))
               .Select(x => new WineQualitySample()
               {
                   FixedAcidity = float.Parse(x[0]),
                   VolatileAcidity = float.Parse(x[1]),
                   CitricAcid = float.Parse(x[2]),
                   ResidualSugar = float.Parse(x[3]),
                   Chlorides = float.Parse(x[4]),
                   FreeSulfurDioxide = float.Parse(x[5]),
                   TotalSulfurDioxide = float.Parse(x[6]),
                   Density = float.Parse(x[7]),
                   Ph = float.Parse(x[8]),
                   Sulphates = float.Parse(x[9]),
                   Alcohol = float.Parse(x[10]),
                   Label = float.Parse(x[11])
               });
        }
    }
}
