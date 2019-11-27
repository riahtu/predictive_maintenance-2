
from utility.Config import *
from utility.Spark import *
from utility.utils import *

class DataCleaning:
    def __init__(self):
        self.spark = get_spark_session("hdsdsdsd")
        self.filename = filename
        self.df_target = read_csv(self.filename)
        # consolidate_csv(data_path)
        self.clean()

    def convert_to_datetime(self, df):
        df = df.withColumn('date_time', 
                   to_date(unix_timestamp(col('date'), 'yyyy-MM-dd').cast("timestamp")))
        return cache(df)

    def clean(self):
        print("converting to timestamp ******************************************************")
        df_temp = self.convert_to_datetime(self.df_target)
        
        # df_temp = df_temp\
        #         .groupBy(month("date").alias("month")).count()
        # df_temp.show()
        print("dropping nan  ******************************************************")
        df_temp = self.drop_nan(self.df_target) 
        print("removing   normalised ******************************************************")
        df_temp = self.remove_normalised_features(df_temp)
        print("populating MFG ******************************************************")
        df_temp = self.populate_MFG(df_temp) 
        print("removing extreme temperatures*********************************************** ")
        df_temp = self.out_of_bounds_temperature(df_temp)
        # print("removing extreme drive days ***************************************")
        # df_temp = self.out_of_bounds_drive_days(df_temp)
        print("WRITING TO FILE ******************************************************")
        write_df_to_file(df_temp)
        df_temp.show()  

    def populate_MFG(self,df):
        match_udf = udf(matchManufacturer)
        df = df\
                .withColumn("MFG",lit(match_udf(split("model", " ")[0])))  
        # write_df_to_file(df_temp)
        return cache(df)
 
    def out_of_bounds_temperature(self, df):
        return cache(df.filter(df.smart_194_raw > 18.0))

    def out_of_bounds_drive_days(self, df):
        return cache(df.filter(df.smart_9_raw > 18.0))

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

        
