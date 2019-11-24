from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest
dataset = [[0, Vectors.dense([0, 0, 1])],
           [0, Vectors.dense([1, 0, 1])],
           [1, Vectors.dense([2, 1, 1])],
           [1, Vectors.dense([3, 1, 1])]]
dataset = spark.createDataFrame(dataset, ["label", "features"])
chiSqResult = ChiSquareTest.test(dataset, 'features', 'label')
print(chiSqResult.select("degreesOfFreedom").collect()[0])