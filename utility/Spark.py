from pyspark.sql import SparkSession
from pyspark.sql.functions import count, when, col

def get_spark_session(nome):
	spark = (SparkSession.builder
		.appName(nome)
		.getOrCreate()   
	)   
    
    # spark.setLogLevel("ERROR")
	return spark