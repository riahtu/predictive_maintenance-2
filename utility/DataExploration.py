import os
from utility.Config import *
from utility.Spark import *
from utility.utils import *


class DataExploration:
    def __init__(self):
        self.spark = get_spark_session("hdsdsdsd")
        self.filename = updated_filename
        self.df_target = read_csv(self.filename)
        # print("df target **********************************")
        # self.df_target.show()
        # self.del_file() 

    def calculate_temperature_average(self):
        # smart_194_raw is temperation SMART Statistic
        df_temp = self.drop_extreme_temp(self.df_target)
        # df_temp.show()
        df_temp = self.change_cols_to_float(df_temp)
        df_temp_avg = df_temp\
                                .groupBy(["MFG","model"])\
                                .agg(round(mean("smart_194_raw"),2).alias("avg_Temp"))\
                                .orderBy("avg_Temp") 
        # model_list = get_col_as_list(df_temp_avg, df_temp_avg.model)
        print("printing avergaes ********************************")
        # df_temp_avg.show()
        self.correlation_temp_failure(df_temp)
        # self.correlation_temp_failure___(df_temp)  

    def drop_extreme_temp(self,df):
        print("printing avergaes ****hjhjhjhjhjhjhjhjhjhjhjhjh****************************")
        return df.filter( (df.smart_194_raw >= 18.0) & (df.smart_194_raw <= 70.0))

    def change_cols_to_float(self,df):
        list_smart_cols = get_smart_stats(df)   
        list_smart_cols.append("failure")
        list_smart_cols.append("capacity_bytes")
        print("this is the list ********************", list_smart_cols)
        for c in list_smart_cols:
            df = df.withColumn(c,col(c).cast(DecimalType(18,0))) # Adjust the decimals you want here.
        print("this is the resulting df after change of cols ************************")
        # df.show()
        return cache(df)

    def correlation_temp_failure(self,df):
        # model_list = ["ST12000NM0007"]
        assembler = VectorAssembler(inputCols=['smart_194_raw'],outputCol="vector_features")
        vectorized_df = assembler.transform(df).select('failure', 'vector_features')
        vectorized_df.show()
        r = ChiSquareTest.test(vectorized_df, "vector_features", "failure").head()
        print("pValues: " + str(r.pValues))
        print("degreesOfFreedom: " + str(r.degreesOfFreedom))
        print("statistics: " + str(r.statistics))

        #    get_all_maker_corr = df.groupBy("model").agg(
        #     corr("smart_194_raw","failure").alias("correlation")).collect() 

        # for row in get_all_maker_corr:
        #     print(row["MFG"],":",row["correlation"])

        # for model in model_list:
        #     print("model ********************************", model)
        #     df = df.filter(col("model") = model)
        #     print(df.stat.corr(model,"failure"))
        #     transformed_df = vector_assembler()
        #     (transformed_df.select(col("failure").alias("label"), col("features"))
        #             .rdd
        #             .map(lambda row: LabeledPoint(row.label, row.features)))
        #     # print(df.stat.chiSqTest())
        # # df.show()
    
    

