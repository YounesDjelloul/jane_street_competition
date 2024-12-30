def analyze_missing_values(df):
    # Convert wide to long format for easier analysis
    df_long = pd.melt(
        df,
        id_vars=['Country', 'Indicator'],
        value_vars=[col for col in df.columns if col.startswith('YR')],
        var_name='Year',
        value_name='Value'
    )

    # Overall missing values by indicator and country
    print("Missing Values Analysis by Indicator and Country:\n")
    pivot_missing = df_long.pivot_table(
        values='Value',
        index='Country',
        columns='Indicator',
        aggfunc=lambda x: x.isna().mean() * 100  # Percentage of missing values
    )

    print("Percentage of missing values for each indicator by country:")
    print(pivot_missing.round(2))
    print("\n" + "=" * 80 + "\n")

    # Missing values summary by indicator
    print("Overall missing values by indicator:")
    indicator_missing = df_long.groupby('Indicator')['Value'].apply(
        lambda x: (x.isna().mean() * 100).round(2)
    ).sort_values(ascending=False)
    print(indicator_missing)
    print("\n" + "=" * 80 + "\n")

    # Missing values summary by country
    print("Overall missing values by country:")
    country_missing = df_long.groupby('Country')['Value'].apply(
        lambda x: (x.isna().mean() * 100).round(2)
    ).sort_values(ascending=False)
    print(country_missing)
    print("\n" + "=" * 80 + "\n")

    # Missing values by year
    print("Missing values by year:")
    year_missing = df_long.groupby('Year')['Value'].apply(
        lambda x: (x.isna().mean() * 100).round(2)
    ).sort_values(ascending=False)
    print(year_missing)


df = pd.read_csv(train_path)

print("Shape of dataset:", df.shape)
print("Columns in dataset:")
print(df.columns)

print("\nBasic Statistics:")
print(df.describe())

missing_values = df.isna().sum()
print("\nMissing values:")
print(missing_values)

target_counts = df["target"].value_counts()
sns.barplot(x=target_counts.index, y=target_counts.values)
plt.title("Target Variable Distribution")
plt.xlabel("Target (0 = No Default, 1 = Default)")
plt.ylabel("Count")
plt.show()

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlations = df[numeric_cols].corr()


def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers


outliers = detect_outliers_iqr(df, 'Total_Amount')
print(f"Outliers in 'Total_Amount':\n{outliers}")

sns.histplot(df["Total_Amount"], bins=50, kde=True)
plt.title("Distribution of Total Loan Amount")
plt.xlabel("Total Amount")
plt.ylabel("Frequency")
plt.show()

sns.boxplot(x=df["target"], y=df["duration"])
plt.title("Loan Duration vs Target")
plt.xlabel("Target")
plt.ylabel("Duration (days)")
plt.show()

sns.boxplot(x=df["target"], y=df["Lender_portion_Funded"])
plt.title("Lender Portion Funded vs Target")
plt.xlabel("Target")
plt.ylabel("Lender Portion Funded (%)")
plt.show()
