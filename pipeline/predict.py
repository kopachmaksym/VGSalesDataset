import pickle
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


import columns

# read train data
ds = pd.read_csv("../data/new_input.csv")
print('new data size', ds.shape)

categorical_columns = ds.select_dtypes(include=['object']).columns

map_dicts = dict()
for column in categorical_columns:
    ds[column] = ds[column].astype('category')
    map_dicts[column] = dict(zip(ds[column], ds[column].cat.codes))
    ds[column] = ds[column].cat.codes


# Define target and features columns
X = ds[columns.X_columns]

# load the model and predict
rf = pickle.load(open('../models/finalized_model.sav', 'rb'))

y_pred = rf.predict(X)

ds['Global_Sales_pred'] = rf.predict(X)
ds.to_csv('../data/prediction_results.csv', index=False)