import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)

print("Shape of the training data:", train.shape)
print("\nData types of each column:\n", train.dtypes)
print("\nMissing values per column:\n", train.isnull().sum())
print("\nPercentage of missing values per column:\n", (train.isnull().sum() / len(train)) * 100)

print("\nDescriptive statistics:\n", train.describe())

numerical_cols = train.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('responder_6')
numerical_cols.remove('date_id')
numerical_cols.remove('time_id')
numerical_cols.remove('symbol_id')

train[numerical_cols].hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.show()

plt.figure(figsize=(20, 15))
sns.boxplot(data=train[numerical_cols])
plt.xticks(rotation=90)
plt.title('Boxplots of Numerical Features')
plt.tight_layout()
plt.show()

correlation_matrix = train[numerical_cols + ['responder_6']].corr(method='spearman')

print("\nCorrelations with responder_6:\n", correlation_matrix['responder_6'].sort_values(ascending=False))

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

print("\nUnique values in date_id:", train['date_id'].nunique())
print("Unique values in time_id:", train['time_id'].nunique())
print("Unique values in symbol_id:", train['symbol_id'].nunique())

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, ax in enumerate(axes.flatten()):
    if i < 9:
        sns.histplot(train[f'responder_{i}'], ax=ax, kde=True)
        ax.set_title(f'responder_{i}')
plt.tight_layout()
plt.show()

train.set_index(['date_id', 'time_id'], inplace=True)
train['responder_6'].plot(figsize=(15, 6))
plt.title('responder_6 over Time')
plt.show()
train.reset_index(inplace=True)

print("\nValue counts for responder_6:\n", train['responder_6'].value_counts())
sns.histplot(train['responder_6'])
plt.title('Distribution of responder_6')
plt.show()