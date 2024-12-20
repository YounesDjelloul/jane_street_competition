import polars as pl

df = pl.read_parquet("./dataset/part-0.parquet")

print(df.select(df.columns[:10]).head())
