# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 08:48:46 2019

@author: Administrator
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


def matchManufacturer(substring):
    
#  print(type(substring))
 manufacturers = {
  "HU": "HGST",
  "ST": "Seagate",
  "WD": "WDC",
  "MD": "TOSHIBA",
  "HM": "HGST",
  "HD": "HGST",
  "HGST": "HGST",
  "TOSHIBA": "TOSHIBA",
  "SAMSUNG": "SAMSUNG",
  "Samsung": "SAMSUNG"
  }
 manufacturer = ''
 for k, v in manufacturers.items():
#    print("string " + substring) 
   if substring.startswith(k):
     manufacturer = manufacturers.get(k, "")
#      print ("starts withhh " + substring)
#      manufacturer = manufacturers.get(k, "")
     return manufacturer
   elif substring == v:
#      manufacturer = manufacturers.get(k, "")
#      print("equal to " + substring)
     return substring
  
pwd = r'C:\Users\Administrator\Documents\test'
os.chdir(pwd) 
extension = 'csv'
#j = 0
for i in glob.glob('*.{}'.format(extension)):
    j = 0
    df = pd.read_csv(i)
    df['length']  = 0
    df['MFG'] = "" 
    df['split'] = ""
    print("iteration" , i) 
    for modell in df['model']:
        print("iteration" , j)
        df['length'][j] = len(modell.split(' '))
        if df['length'][j] > 1:
            substring = modell.split()[0]
        
        else: 
            substring = str(modell)
            df['split'][j] = "no "     
        df['MFG'][j] = matchManufacturer(substring)
        j+=1
    df.to_csv(i, index=False, encoding='utf-8-sig')






#df = pd.read_csv('harddrive.csv')
#print(df.shape)
#print(df.columns)
#print(df['MFG'].value_counts()) 

#mfg_rows = df[df['MFG'].isnull()]
#print(mfg_rows)
#mfg_rows.to_csv("final.csv")
#df = df.drop(['length', 'split'], axis=1)
#df = df[df.model_new != "model_new"]
#df.to_csv("final.csv", header=False,index = False, mode = 'a')
##print(df)
## 


#print(df['Datetime'].unique())
#print(df.groupby(['MFG', 'model_new']).agg({'MFG':'count'}))
#print(pd.DataFrame(df.groupby('MFG')['model_new'].nunique()))
#print(df.groupby('model_nemodel_w')['MFG'])
#print(df.groupby(['model_new']))