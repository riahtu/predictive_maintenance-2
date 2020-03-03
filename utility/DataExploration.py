import os
from utility.Config import *
from utility.Spark import *
from utility.utils import *


class DataExploration:
    def __init__(self):
        self.spark = get_spark_session("hdsdsdsd")
        self.filename = append_id(filename,"cleaning")
        self.df_target = read_csv(self.filename)

        self.calculate_temperature_average()
        # print("df target **********************************")
        # self.df_target.show()
        # self.del_file() 

    def calculate_temperature_average(self): 
        # smart_194_raw is temperation SMART Statistic
        df_temp = self.drop_extreme_temp(self.df_target)
        # df_temp.show()
        df_temp = change_cols_to_float(df_temp)
        print(df_temp.dtypes, "explore types*******************")
        print("****************************************************************")
        write_df_to_file(df_temp, "exploration")
        df_temp_avg = df_temp\
                                .groupBy(["MFG","model"])\
                                .agg(round(mean("smart_194_raw"),2).alias("avg_Temp"))\
                                .orderBy("avg_Temp") 
        model_list = get_col_as_list(df_temp_avg, df_temp_avg.model)
        print("printing avergaes ********************************")
        correlation = self.correlation_temp_failure(df_temp, model_list)
        print("WRITING TO FILE ******************************************************")
        write_df_to_file(correlation, "correlation")
        correlation.show()
        return correlation
       
    def drop_extreme_temp(self,df):
        print("printing avergaes ****hjhjhjhjhjhjhjhjhjhjhjhjh****************************")
        return df.filter((df.smart_194_raw >= 18.0))

    

    def correlation_temp_failure(self,df, model_list):
        # model_list = ["TOSHIBA MD04ABA400V"] 
        print("corelation temppppppppp***************")
        new_df = self.create_dataframe()
        # new_df.show()
        for model in model_list:
            print("printing for model ********************** ", str(model))
            df_temp = df.filter(df.model == str(model))
            drive_age = self.get_drive_age_per_model(df_temp)
            print("correlation **************************", df_temp.stat.corr("smart_194_raw","failure"))
            stat = df_temp.stat.corr("smart_194_raw","failure")
            assembler = self.vector_assembler(df_temp)
            p_value = self.get_p_value(assembler)
            print("thisis the value of pv*************** ", p_value)
            significance = str(self.significance(p_value))
            statistic = self.get_statistic(assembler)
            num_dead = self.num_failed_per_model(df_temp,str(model))
            print("num deadddd real () ", num_dead)
            num_alive = self.num_alive_per_model(df_temp, str(model))
            print("num alivedd real () ", num_alive)
            new_df = self.populate_df(new_df, model, stat, significance, p_value, num_dead, num_alive, drive_age)
        new_df = new_df.orderBy("p_value")
        return new_df
        

    def get_drive_age_per_model(self, df):
        df = df.select(((mean(df.smart_9_raw)/24)/365).alias("drive_age"))
        # df.show()
        return df.select("drive_age").rdd.flatMap(lambda x: x).collect()[0] 
       
    def create_dataframe(self):
        schema = StructType([
        StructField("Model",StringType(),True),
        StructField("stat",DoubleType(),True),
        StructField("Significance",StringType(),True),
        StructField("p_value",DoubleType(),True),
        StructField("Num_dead",IntegerType(),True),
        StructField("Num_alive",IntegerType(),True),
        StructField("Drive_Age (years)",DoubleType(),True)
        ])
        return spark.createDataFrame([], schema)
    
    def populate_df(self,new_df, model, stat, significance, p_value, num_dead, num_alive, drive_age):
        newRow = spark.createDataFrame([(model, stat, significance, p_value, num_dead, num_alive,drive_age)])
        # print("printing new row  ************** ", newRow)
        new_df = new_df.union(newRow)
        # new_df.show()
        return new_df 


    def num_failed_per_model(self, df, model):
        # print("num dead ***************" , df.filter((df.model == model) & (df.failure == 1) ).count())
        df = df.filter((df.model == model) & (df.failure == 1) )
        df = df.select(countDistinct("serial_number").alias("distinct_model"))
        col_list = get_col_as_list(df,df.distinct_model)
        # print("printing count  dead ****************** ", col_list[0])
        # df.show()

        return  col_list[0]

    def num_alive_per_model(self, df, model):
        # print("num alive ******************** ", df.filter((df.model == model) & (df.failure == 0) ).count())
        df = df.filter((df.model == model) & (df.failure == 0) )
        # print("printing count  alive ******************")
        df = df.select(countDistinct("serial_number").alias("distinct_model"))
        col_list = get_col_as_list(df,df.distinct_model)
        # print("printing count  alive ****************** ", col_list[0])
        return col_list[0]


    def significance(self,p_value):
        # print("significance : ", p_value <= 0.05)
        return (p_value <= 0.05)

    def vector_assembler(self,df):
        assembler = VectorAssembler(inputCols=['smart_194_raw'],outputCol="vector_features")
        vectorized_df = assembler.transform(df).select('failure', 'vector_features')
        r = ChiSquareTest.test(vectorized_df, "vector_features", "failure").head()
        return r

    def get_statistic(self, r):
        print("statistic :", str(r.statistics))
        return str(r.statistics)

    def get_p_value(self,r):
        print("pValues: " , r.pValues[0], float(r.pValues[0]))
        return float(r.pValues[0])
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
    
    

