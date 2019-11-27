from utility.Config import *
from utility.Spark import *
from pyspark.sql.types import *
import sys, os, glob
import pandas as pd
from datetime import datetime
from pyspark.mllib.stat import Statistics
from  pyspark.sql.functions import * 
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.stat import ChiSquareTest

spark = get_spark_session("")
spark.sparkContext.setLogLevel('ERROR')

def drop_nan(df):
    aggregated_row = df.select([(count(when(col(c).isNull(), c))/df.count()).alias(c) for c in df.columns]).collect()
    aggregated_dict_list = [row.asDict() for row in aggregated_row]
    aggregated_dict = aggregated_dict_list[0]  
    col_null_g_75p=list({i for i in aggregated_dict if aggregated_dict[i] > 0.75})
    df = df.drop(*col_null_g_75p).cache()
    return df

def cache(df):
    return df.cache() 

def mean_udf(v):
    return round(v.mean(), 2)

def read_csv(filename):
    return cache(spark.read.csv(filename, header="true"))

def write_df_to_file(df):
    df.write.csv(updated_filename, header="true")
    # del_file(filename)
    # filename = updated_filename
    # return read_csv(filename)
def get_smart_stats(df):
    return [i for i in df.columns if i[:len('smart')]=='smart']

def get_col_as_list(df ,col_):
    return df.select(col_).rdd.flatMap(lambda x: x).collect()

def vector_assembler(df):
    smart_list  = get_smart_stats(df)
    assembler = VectorAssembler(inputCols=smart_list,outputCol="features")
    return assembler.transform(df)

def del_file():
    os.remove(filename)

def matchManufacturer(manufacturer):
    n={"HU":"HGST","Hitachi":"HGST","ST":"Seagate","WD":"WDC","MD":"Toshiba","HM":"HGST","HD":"HGST","HGST":"HGST","TOSHIBA":"TOSHIBA","SAMSUNG":"SAMSUNG","Samsung":"Samsung"}
    for k in n:
        if str(manufacturer).startswith(k):
            return n[k]
        elif manufacturer == k:
            return n[k] 
        