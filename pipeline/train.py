import pickle
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# custom files
import model_best_hyperparameters
import columns

# read train data
ds = pd.read_csv("../data/train.csv")

categorical_columns = ds.select_dtypes(include=['object']).columns

map_dicts = dict()
for column in categorical_columns:
    ds[column] = ds[column].astype('category')
    map_dicts[column] = dict(zip(ds[column], ds[column].cat.codes))
    ds[column] = ds[column].cat.codes

# Define target and features columns
X = ds[columns.X_columns]
y = ds[columns.y_column]

# Let's say we want to split the data in 90:10 for train:test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.9)

# Building and train Random Forest Model
rf = RandomForestRegressor(**model_best_hyperparameters.params)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_absolute_percentage_error(y_test, y_pred))

filename = '../models/finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))