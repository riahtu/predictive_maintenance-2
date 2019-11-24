from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest
from utility.Spark import *

spark = get_spark_session("hdsdsdsd")
spark.sparkContext.setLogLevel('ERROR')
from pyspark.ml.feature import RFormula

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
dataset = [[Vectors.dense([1, 0, 0, -2])],
           [Vectors.dense([4, 5, 0, 3])],
           [Vectors.dense([6, 7, 0, 8])],
           [Vectors.dense([9, 0, 0, 1])]]
dataset = spark.createDataFrame(dataset, ['features'])
dataset.show()
pearsonCorr = Correlation.corr(dataset, 'features', 'pearson').collect()[0][0]
print(str(pearsonCorr).replace('nan', 'NaN'))

spearmanCorr = Correlation.corr(dataset, 'features', method='spearman').collect()[0][0]
print(str(spearmanCorr).replace('nan', 'NaN'))
