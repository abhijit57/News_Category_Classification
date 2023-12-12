import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# Converting the input file from txt format to csv format
with open('News_Category_Dataset_v3.txt', 'r') as f:
    data = f.readlines()
list_of_dicts = []
for d in data:
    list_of_dicts.append(eval(d))
df = pd.DataFrame(columns=['link', 'headline', 'category', 'short_description', 'authors', 'date'])
df = df.from_records(list_of_dicts)


print('Data Consistency Checks....... ','\n')
# Removing the Non UTF-8 characters
df_encoded = df.applymap(lambda x: x.encode('utf-8', 'ignore').decode('utf-8'))
# Check for missing values
missing_values = df_encoded.isnull().sum()
print("Missing Values:")
print(missing_values)
# Check date and time consistency
try:
    df_encoded['date'] = pd.to_datetime(df_encoded['date'])
    print("\nDate and Time Consistency: No issues found.")
except ValueError as e:
    print(f"\nDate and Time Consistency Issue: {e}")
# Check for duplicates
duplicates = df_encoded.duplicated().sum()
print("\nDuplicates:", duplicates)
if duplicates:
    df_encoded.drop_duplicates(keep='first', inplace=True)
    duplicates = df_encoded.duplicated().sum()
    print('Removing the duplicates found..')
    print('Updated duplicate count: ', duplicates)


# Concatenating columns and dropping unnecessary columns
data = pd.DataFrame()
data['news'] = df_encoded['headline'] + '. ' + df_encoded['short_description']
data['category'] = df_encoded['category']


# Train Test Val Split
train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=np.array(data['category']))
test, val = train_test_split(test, test_size=0.4, random_state=42, stratify=np.array(test['category']))
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
val.reset_index(drop=True, inplace=True)
train.to_csv('news_train_data.csv', index=False)
test.to_csv('news_test_data.csv', index=False)
val.to_csv('news_val_data.csv', index=False)

