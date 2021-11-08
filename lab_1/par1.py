import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sns 
import matplotlib.pyplot as plt 

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv" 
test = pd.read_csv(test_url) 

#For the train set 
train.isna().head()  
# For the test set 
test.isna().head()  
#Let's get the total number of missing values in both datasets. 
print("*****In the train set*****") 
print(train.isna().sum()) 
print("\n") 
print("*****In the test set*****") 
print(test.isna().sum())

#Fill missing values with mean column values in the train dataset
train.fillna(train.mean(), inplace=True)
#Fill missing values with mean column values in the test dataset
test.fillna(test.mean(), inplace=True)

print(test.isna().sum())

g = sns.FacetGrid(train, col='Survived') 
g.map(plt.hist, 'Age', bins=20)
