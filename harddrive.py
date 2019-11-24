import sys
print (sys.path)
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl   
import stats
import statsmodels.api as sm
import warnings

from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')
import os
import glob
from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from sqlalchemy import create_engine
import pyspark.sql.functions as F
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
   
from pandas.plotting import scatter_matrix

filename = 'harrdrive_edited.csv'
filename_edited ='harrdrive_manufacturers.csv'
#df = pd.read_csv(filename,chunksize=10000)  

def matchManufacturer(substring):
  manufacturers = {
  "HU": "HGST",
  "ST": "Seagate",
  "WD": "WDC",
  "MD": "Toshiba",
  "HM": "HGST",
  "HD": "HGST",
  "HGST": "HGST",
  "TOSHIBA": "TOSHIBA",
  "SAMSUNG": "SAMSUNG",
  "Samsung": "Samsung"
  }
  manufacturer = ''
  substring = substring.strip()
  for k, v in manufacturers.items():

    if substring.startswith(k):
      manufacturer = manufacturers.get(k, "")
#      print ("starts withhh " + substring)
#      manufacturer = manufacturers.get(k, "")
      return manufacturer
    elif substring == v:
#      manufacturer = manufacturers.get(k, "")
#      print("equal to " + substring)
      return substring 

def failure_rates():
    try:
        connection = pymysql.connect(host='localhost',
                             database='thesis_harddrive',
                             user='root',
                             password='')
        if connection.open:
          db_Info = connection.get_server_info()
          print("Connected to MySQL database... MySQL Server version on ",db_Info)
          cursor = connection.cursor()
          cursor.execute("select * from failure_rates order by model;")
          record = cursor.fetchone()
          print ("Your connected to - ", record)
    finally:
        if(connection.open):
          cursor.close()
          connection.close()
          print("MySQL connection is closed")
    

def consolidate_csv():
    pwd = r'C:\Users\Administrator\Documents\done'
    os.chdir(pwd)
    extension = 'csv'
    processed_files = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in processed_files])
    
#export to csv
    combined_csv.to_csv( "harddrive_combined_2018q1-2.csv", index=False, encoding='utf-8-sig')

#pass new file name for feature extraction (combines files)
def feature_extraction_constants(new_filename):
    data = pd.read_csv(new_filename) 
    
    train_features, test_features, train_labels, test_labels=train_test_split(  
            data.drop(labels=['failure','date','model', 'MFG','Datetime','serial_number', 'model_new', 'length','split','Unnamed: 0',  'Unnamed: 0.1',  'Unnamed: 0.1.1'], axis=1),
            data['failure'],
            test_size=0.2,
            random_state=41)
    constant_filter = VarianceThreshold(threshold=0)  
    constant_filter.fit(train_features)  
    len(train_features.columns[constant_filter.get_support()])  
    constant_columns = [column for column in train_features.columns  
                    if column not in train_features.columns[constant_filter.get_support()]]

    train_features = train_features.drop(constant_columns,axis=1)
    test_features = test_features.drop(constant_columns,axis=1)
    
#    create quasi constant filter
    qconstant_filter = VarianceThreshold(threshold=0.03)  
    qconstant_filter.fit(train_features)  

    len(train_features.columns[qconstant_filter.get_support()])
    qconstant_columns = [column for column in train_features.columns  
                    if column not in train_features.columns[qconstant_filter.get_support()]] 

#    drop least informative columns from original dataset
    data = data.drop(['Unnamed: 0.1',  'Unnamed: 0.1.1', 'split','length'], axis=1)
    data = data.drop(constant_columns, axis=1)
    data = data.drop(qconstant_columns, axis=1)
    data.to_csv("harddrive_features.csv", header=True, index = False, mode = 'a')
    
def feature_extraction_duplicates(new_filename):
    data = pd.read_csv(new_filename)  
    print(data.shape)

    train_features, test_features, train_labels, test_labels=train_test_split(  
            data.drop(labels=['failure','date','model', 'MFG','Datetime','serial_number', 'model_new' ], axis=1),
            data['failure'],
            test_size=0.2,
            random_state=41)
    train_features_T = train_features.T  
    train_features_T.shape  
    print("here")
    print(train_features_T.duplicated().sum())  
    print("after print")
    unique_features = train_features_T.drop_duplicates(keep='first').T  
    print(unique_features.shape)  
    duplicated_features = [dup_col for dup_col in train_features.columns if dup_col not in unique_features.columns]  
    print(duplicated_features) 

def clean():
    
    filename = r"C:\Users\Administrator\Documents\thesis\harddrive_final.csv"
    df = pd.read_csv(filename)  
    
#drop nan values and columns with more than 75% nan values
    df = df.dropna(axis='columns', how='all')
    df = df.dropna(axis='columns', thresh = int(0.75 * len(df)))
    
    #fill remaining missing columns with mean of column
    colnames_numerics_only = df.select_dtypes(include=[np.number]).columns.tolist()
    for column in colnames_numerics_only:
        df[column] = df[column].fillna(df.loc[:,column].mean())
    
    #change date type to  datetime 
    df['Datetime'] = pd.to_datetime(df['date'])
    
    df.to_csv(filename,  index=False)

def load_intn_sql_lite():
    filename= r"harddrive.csv"
    csv_database = create_engine('sqlite:///csv_database.db')
    i = 0
    j = 1
    for df in pd.read_csv(filename, chunksize=100000, iterator=True):
        df = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) 
        df.index += j
        i+=1
        df.to_sql('drives', csv_database, if_exists='append')
        j = df.index[-1] + 1

        
    
def variable_correlation(new_filename):
    df = pd.read_csv(new_filename)
    print(df.head())
    print(df.columns)
    df = df.drop(['date', 'Datetime','model','model_new'], axis=1)
    raw_list = [col for col in df.columns if 'normalized' in col]
    print(raw_list)
    df = df.drop(raw_list,axis = 1)
    
    ##Using Pearson Correlation
    plt.figure(figsize=(12,10))
    cor = df.corr()  
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()      
    
    
    #Correlation with output variable
    cor_target = abs(cor["failure"]) 
    
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.5]
    print(relevant_features)

def manufacturer_distribution():
    
    df = sqlContext.read.load('harddrive.csv', delimiter=';',format='com.databricks.spark.csv', header='true', inferSchema='true')
    dg = df.groupBy("MFG").agg(F.countDistinct("model_new"))
#*****************************************************************************************************    
#load_intn_sql_lite()


#clean()
#consolidate_csv()
#feature_extraction_constants(r"C:\Users\Administrator\Documents\harddrive_combined.csv")
#feature_extraction_duplicates(r"C:\Users\Administrator\Documents\harddrive.csv")
#variable_correlation(r"C:\Users\Administrator\Documents\harddrive.csv")

#consolidate_csv()
#calculate failure rates mysql 













#df = pd.read_csv("harddrive_combined.csv") 


#drop nan values and columns with more than 75% nan values

#df = df.dropna(axis='columns', how='all')
#df = df.dropna(axis='columns', thresh = int(0.75 * len(df)))

#fill remaining missing columns with mean of column
#colnames_numerics_only = df.select_dtypes(include=[np.number]).columns.tolist()
#for column in colnames_numerics_only:
#    df[column] = df[column].fillna(df.loc[:,column].mean())

#chnage date type to  datetime 
#df['Datetime'] = pd.to_datetime(df['date'])

#strip leading and trailing spaces in model number
#df['model'] = df['model'].str.strip() 

#replace 9Hitachi with HGST
#df['model_new'] = df['model'].str.replace('Hitachi','HGST')
 
#dfread = dfread.drop(['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1'], axis=1)

#read file in chunck to add manufacturer column
#dfread = pd.read_csv(r"C:\Users\Administrator\Documents\harddrive.csv", iterator=True, chunksize=700000) 


#df= pd.read_csv(r"C:\Users\Administrator\Documents\null.csv")
#i = 0
##for df in dfread: 
###while True:
###  df = dfread.get_chunk(100) 
###  print(df.size + "this si the size")
##
##  df.drop(['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1'], axis=1) 
#df['length']  = 0
#df['MFG'] = "" 
#df['split'] = ""
#for modell in df['model_new']:
#    print("iteration " , i)
#    print(modell)
#    df['length'][i] = len(modell.split(' '))
#    if df['length'][i] > 1:
#        substring = modell.split()[0]
#    #        str(df['model_new'][i].map(lambda x: x.split(' ')[0]))
#        df['split'][i] = substring
#    #        
#    else: 
#        substring = str(modell)
#        df['split'][i] = "no "
#    #  #     
#    df['MFG'][i] = matchManufacturer(substring.strip()) 
#    #
#    i+=1
#df.to_csv("filled_mfg.csv",index = False, mode = 'a',header=True) 
#


