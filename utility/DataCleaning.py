
from utility.Config import *
from utility.Spark import *
from utility.utils import *

class DataCleaning:
    def __init__(self):
        self.spark = get_spark_session("hdsdsdsd")
        self.filename = filename
        # self.df_target = read_csv(self.filename)
        consolidate_csv(data_path)
        # self.clean()

    def convert_to_datetime(self, df):
        df = df.withColumn('date_time', 
                   to_date(unix_timestamp(col('date'), 'yyyy-MM-dd').cast("timestamp")))
        df.show()
        return cache(df)

    def clean(self):
        df_temp = self.convert_to_datetime(self.df_target)
        print("******************************************************")
        df_temp = df_temp\
                .groupBy(month("date").alias("month")).count()
        df_temp.show()
        # df_temp = self.drop_nan(self.df_target) 
        # df_temp = self.convert_to_datetime(df_temp)
        # df_temp = self.remove_normalised_features(df_temp)
        # df_temp = self.populate_MFG(df_temp)
        # write_df_to_file(df_temp)
        # df_temp.show() 

    def populate_MFG(self,df):
        match_udf = udf(matchManufacturer)
        df = df\
                .withColumn("MFG",lit(match_udf(split("model", " ")[0])))  
        # write_df_to_file(df_temp)
        return cache(df)

    def drop_nan(self, df):
        aggregated_row = df.select([(count(when(col(c).isNull(), c))/df.count()).alias(c) for c in df.columns]).collect()
        aggregated_dict_list = [row.asDict() for row in aggregated_row]
        aggregated_dict = aggregated_dict_list[0]  
        col_null_g_75p=list({i for i in aggregated_dict if aggregated_dict[i] > 0.75})
        df = df.drop(*col_null_g_75p)
        return cache(df)

    def remove_normalised_features(self,df):
        raw_list = list({col for col in df.columns if 'normalized' in col})
        print(raw_list)
        df = df.drop(*raw_list)
        return cache(df)

        
