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