from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset from the uploaded CSV file
file_path = '../data/vgsales.csv'
data = pd.read_csv(file_path)
# Splitting the data into train and test sets in a 90:10 ratio
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# Output the sizes of the train and test datasets
print(train_data.shape, test_data.shape)

train_data_path = '../data/train.csv'
test_data_path = '../data/new_input.csv'

train_data.to_csv(train_data_path, index=False)
test_data.to_csv(test_data_path, index=False)