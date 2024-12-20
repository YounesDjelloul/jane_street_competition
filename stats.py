def get_dataset_stats(df_lazy):
    print("Basic statistics for responder_6:")
    stats = df_lazy.select([
        pl.col("responder_6").mean().alias("mean"),
        pl.col("responder_6").std().alias("std"),
        pl.col("responder_6").min().alias("min"),
        pl.col("responder_6").max().alias("max"),
        pl.col("responder_6").quantile(0.25).alias("25%"),
        pl.col("responder_6").quantile(0.50).alias("50%"),
        pl.col("responder_6").quantile(0.75).alias("75%")
    ]).collect()
    print(stats)

    # For correlations with target, we can calculate them one by one
    # Let's get top correlated features
    feature_cors = []
    for feature in [f"feature_{i:02d}" for i in range(79)]:
        cor = df_lazy.select(
            pl.corr("responder_6", feature).alias("correlation")
        ).collect().item()
        feature_cors.append((feature, cor))

    # Sort and print top correlations
    feature_cors.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\nTop 10 correlated features with responder_6:")
    for feature, cor in feature_cors[:10]:
        print(f"{feature}: {cor:.4f}")

    # Get null counts
    print("\nNull counts:")
    null_counts = df_lazy.select([
        pl.col("*").null_count()
    ]).collect()
    print(null_counts)

    # plt.figure(figsize=(10, 6))
    # plt.hist(df_pd['responder_6'], bins=50)
    # plt.title('Distribution of responder_6')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.show()