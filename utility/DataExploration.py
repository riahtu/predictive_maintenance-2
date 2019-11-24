import os
from utility.Config import *
from utility.Spark import *
from utility.utils import *


class DataExploration:
    def __init__(self):
        self.spark = get_spark_session("hdsdsdsd")
        self.filename = updated_filename
        self.df_target = read_csv(self.filename)
        # self.del_file() 

    def calculate_temperature_average(self):
        # smart_194_raw is temperation SMART Statistic
        # self.df_target = self.df_target\
        #                         .groupBy(["MFG","model"])\
        #                         .agg(round(mean("smart_194_raw"),2).alias("avg_Temp"))\
        #                         .orderBy("avg_Temp") 
        # self.df_target.show()
        self.correlation_temp_failure(self.df_target) 
  
    def correlation_temp_failure(self,df):
        get_all_maker_corr = df.groupBy("model").agg(
            corr("smart_194_raw","failure").alias("correlation")).collect() 

        for row in get_all_maker_corr:
            print(row["MFG"],":",row["correlation"])

        
    

