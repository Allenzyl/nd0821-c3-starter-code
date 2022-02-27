import pandas as pd
#import os
#print(os.listdir())
data = pd.read_csv("data/census.csv")

data = data.drop_duplicates().reset_index()
data.info()
data.columns = data.columns.str.replace(' ', '')
data.to_csv("data/census_cleaned.csv")