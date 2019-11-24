from utility.Config import *
from utility.Spark import *
from pyspark.sql.types import DateType
import sys, os, glob
import pandas as pd
from datetime import datetime
from pyspark.mllib.stat import Statistics
from  pyspark.sql.functions import * 
from pyspark.ml.regression import GeneralizedLinearRegression


spark = get_spark_session("")
spark.sparkContext.setLogLevel('ERROR')


def consolidate_csv(data_path):
    
    os.chdir(os.path.join(sys.path[0], data_path)) 
    extension = 'csv'
    raw_files = [i for i in glob.glob('*.{}'.format(extension))]

    combined_csv = pd.concat([pd.read_csv(f) for f in raw_files])
    combined_csv.to_csv( "harddrive.csv", index=False, encoding='utf-8-sig')

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

def del_file():
    os.remove(filename)

def matchManufacturer(manufacturer):
    n={"HU":"HGST","Hitachi":"HGST","ST":"Seagate","WD":"WDC","MD":"Toshiba","HM":"HGST","HD":"HGST","HGST":"HGST","TOSHIBA":"TOSHIBA","SAMSUNG":"SAMSUNG","Samsung":"Samsung"}
    for k in n:
        if str(manufacturer).startswith(k):
            return n[k]
        elif manufacturer == k:
            return n[k] 
        