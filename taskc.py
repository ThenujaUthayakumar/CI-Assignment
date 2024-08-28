import pandas as pd

df = pd.read_csv('train.csv')

print(df.head())
print(df.columns)

print(f"Dataset Shape: {df.shape}")

print(df.isnull().sum())

print(df.describe(include='all'))

#================================================
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='class')
plt.title('Distribution of Edible vs Poisonous Mushrooms')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

class_distribution = df['class'].value_counts(normalize=True) * 100
print(class_distribution)

#======================================
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='cap_shape', hue='class')
plt.title('Cap Shape Distribution by Class')
plt.xlabel('Cap Shape')
plt.ylabel('Count')
plt.legend(title='Class', labels=['Edible', 'Poisonous'])
plt.show()

#===========================================
print("Columns in DataFrame:", df.columns)

df.columns = df.columns.str.strip()

if 'cap-shape' in df.columns and 'class' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='cap-shape', hue='class')
    plt.title('Cap Shape Distribution by Class')
    plt.xlabel('Cap Shape')
    plt.ylabel('Count')
    plt.legend(title='Class', labels=['Edible', 'Poisonous'])
    plt.show()
else:
    print("Error: Columns 'cap_shape' or 'class' are not in the DataFrame.")

#=====================================
print("Columns in DataFrame:", df.columns)

print(df.head())

df.columns = df.columns.str.strip()

if 'cap-color' in df.columns and 'class' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='cap-color', hue='class')
    plt.title('Cap Color Distribution by Class')
    plt.xlabel('Cap Color')
    plt.ylabel('Count')
    plt.legend(title='Class', labels=['Edible', 'Poisonous'])
    plt.show()
else:
    print("Error: Columns 'cap_color' or 'class' are not in the DataFrame.")
    print("Available columns:", df.columns)

#=================================
print("Columns in DataFrame:", df.columns)
print(df.head())
df.columns = df.columns.str.strip()

if 'does-bruise-or-bleed' in df.columns and 'class' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='does-bruise-or-bleed', hue='class')
    plt.title('Bruises Distribution by Class')
    plt.xlabel('Does Bruise or Bleed')
    plt.ylabel('Count')
    plt.legend(title='Class', labels=['Edible', 'Poisonous'])
    plt.show()
else:
    print("Error: Columns 'does-bruise-or-bleed' or 'class' are not in the DataFrame.")
    print("Available columns:", df.columns)

#================================
print("Columns in DataFrame:", df.columns)
print(df.head())

df.columns = df.columns.str.strip()

plt.figure(figsize=(12, 8))
sns.catplot(x='cap_shape', hue='class', col='odor', data=df, kind='count', height=4, aspect=0.7)
plt.subplots_adjust(top=0.9)
plt.suptitle('Cap Shape vs Odor by Class')
plt.show()

#===============================
plt.figure(figsize=(12, 8))
sns.catplot(x='gill-color', hue='class', col='cap-shape', data=df, kind='count', height=4, aspect=0.7)
plt.subplots_adjust(top=0.9)
plt.suptitle('Gill Color vs Cap Shape by Class')
plt.show()


#======================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import psutil

def process_chunk(chunk):
    for col in chunk.select_dtypes(include=['object']).columns:
        chunk[col] = chunk[col].astype('category')
    return chunk

chunk_size = 100000  
chunks = pd.read_csv('train.csv', chunksize=chunk_size)

df_list = []
for chunk in chunks:
    chunk = process_chunk(chunk)
    df_list.append(chunk)
df = pd.concat(df_list, ignore_index=True)

print("Optimized Data Types:")
print(df.dtypes)

process = psutil.Process()
print(f"\nMemory Usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

label_encoder = LabelEncoder()
encoded_columns = []

for column in df.select_dtypes(include=['category']).columns:
    encoded_column_name = column + '_encoded'
    df[encoded_column_name] = label_encoder.fit_transform(df[column])
    encoded_columns.append(encoded_column_name)

df_encoded = df[encoded_columns + ['class']].copy()

if df_encoded['class'].dtype.name == 'category':
    df_encoded['class'] = label_encoder.fit_transform(df_encoded['class'])

print("Data types of columns in df_encoded:")
print(df_encoded.dtypes)

df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce') 
print("Check for NaNs in df_encoded:")
print(df_encoded.isna().sum())

df_encoded = df_encoded.dropna()

correlation_matrix = df_encoded.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Encoded Features')
plt.show()

#================================================
features = [col for col in df_encoded.columns if col.endswith('_encoded')]

print("Features available for pairplot:")
print(features)

if features: 
    sns.pairplot(df_encoded[features + ['class']], hue='class', palette='Set1')
    plt.show()
else:
    print("No valid features available for pairplot.")



