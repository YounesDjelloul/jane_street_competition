import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

str1 = '../input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id='
str2 = '/part-0.parquet'
file_paths = [f"{str1}{i}{str2}" for i in range(4)]

dataframes = [pd.read_parquet(file) for file in file_paths[-4:]]
train = pd.concat(dataframes)

train.head()

train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

train_dataset = FinancialDataset(train_df)
val_dataset = FinancialDataset(val_df)

input_size = len(train_dataset.feature_cols)
model = FinancialNN(input_size)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024)

trained_model = train_model(model, train_loader, val_loader, epochs=15, lr=0.0005)