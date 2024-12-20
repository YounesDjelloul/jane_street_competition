import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from stats import get_dataset_stats
from train import FinancialDataset, FinancialNN, train_model

datasets_path = "./dataset/train.parquet/partitions/*.parquet"
df_lazy = pl.scan_parquet(datasets_path)

get_dataset_stats(df_lazy)

dataset = FinancialDataset(datasets_path)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024)

model = FinancialNN()
trained_model = train_model(model, train_loader, val_loader)
