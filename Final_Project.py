import pandas as pd
import numpy as np
import random as rnd
import scipy
import seaborn as sns


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

# print(train_df.columns.values)
# print(train_df.head())
# print(train_df.info())
# print(test_df.info())
print(train_df.isnull().sum())
print(train_df.describe)

train_df.info()
test_df.info()

sns.countplot("Class", data=combine)





