# Part 1 - EDA on Diamonds dataset
# Dataset: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(os.path.dirname(__file__) or ".")

# change this to True if you want plots to pop up on screen
SHOW_PLOTS = False

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
OUT = "output"
FIGS = os.path.join(OUT, "figures")
os.makedirs(FIGS, exist_ok=True)

sns.set_theme(style="whitegrid")


def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, name), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()



# 1) Load data

print("1) LOAD DATASET")
# loading table as object on purpose - will fix dtypes in step 4
df = pd.read_csv(DATA_URL, dtype={"table": "object"})
print(df.head())
print(df.dtypes)
print("Shape:", df.shape)



# 2) Null analysis

print("\n2) NULL VALUE ANALYSIS")
null_counts = df.isnull().sum()
null_pct = (null_counts / df.shape[0]) * 100
null_table = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
null_table = null_table.sort_values("null_pct", ascending=False)
print(null_table)

high_null = null_table[null_table["null_pct"] > 20].index.tolist()
print("Columns with more than 20% nulls:", high_null if high_null else "None")

# fill numeric nulls under 20% with median
for col in df.select_dtypes(include="number").columns:
    if null_pct[col] < 20 and null_counts[col] > 0:
        df[col] = df[col].fillna(df[col].median())



# 3) Duplicates

print("\n3) DUPLICATE DETECTION")
before = df.shape[0]
print("Duplicates found:", df.duplicated().sum())
df = df.drop_duplicates().copy()
after = df.shape[0]
print("Rows removed:", before - after)

# check if null % changed after removing duplicates
null_pct_after = (df.isnull().sum() / df.shape[0]) * 100
print("Null % change per column (should be ~0):")
print((null_pct_after - null_pct).round(4))



# 4) Fix data types

print("\n4) DATA TYPE CORRECTION")
mem_before = df.memory_usage(deep=True).sum()
print("Memory before:", mem_before, "bytes")

# table column came in as object, convert it
df["table"] = pd.to_numeric(df["table"], errors="coerce")

# these string columns repeat a lot, category saves memory
for col in ["cut", "color", "clarity"]:
    df[col] = df[col].astype("category")

mem_after = df.memory_usage(deep=True).sum()
print("Memory after:", mem_after, "bytes")
print("Saved:", mem_before - mem_after, "bytes")
print(df.dtypes)



# 5) Stats and skewness

print("\n5) DESCRIPTIVE STATISTICS + SKEWNESS")
num_cols = df.select_dtypes(include="number").columns.tolist()
print(df[num_cols].describe())

skew_vals = df[num_cols].skew().sort_values(key=lambda s: s.abs(), ascending=False)
print("\nSkewness:")
print(skew_vals)
most_skewed = skew_vals.index[0]
print("Most skewed column:", most_skewed)



# 6) IQR outlier check - price and carat

print("\n6) IQR OUTLIER DETECTION")
outlier_rows = []
for col in ["price", "carat"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out_count = int(((df[col] < lower) | (df[col] > upper)).sum())
    outlier_rows.append({"column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
                         "lower": lower, "upper": upper, "outlier_count": out_count})
outlier_df = pd.DataFrame(outlier_rows)
print(outlier_df.round(4).to_string(index=False))



# 7-12) Visualizations

print("\n7-12) VISUALIZATIONS")

# line plot
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["price"], linewidth=0.8)
plt.title("Price by Row Index")
plt.xlabel("Row Index")
plt.ylabel("Price")
save_plot("line_plot_price.png")

# bar chart - mean price per cut
avg_price = df.groupby("cut", observed=False)["price"].mean().sort_values()
plt.figure(figsize=(8, 4))
plt.bar(avg_price.index.astype(str), avg_price.values)
plt.title("Mean Price by Cut")
plt.xlabel("Cut")
plt.ylabel("Mean Price")
plt.xticks(rotation=15)
save_plot("bar_mean_price_by_cut.png")

# histogram of most skewed column
plt.figure(figsize=(8, 4))
sns.histplot(df[most_skewed], bins=20)
plt.title("Histogram - " + most_skewed)
plt.xlabel(most_skewed)
plt.ylabel("Count")
save_plot("hist_" + most_skewed + ".png")

# scatter
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="carat", y="price", alpha=0.3, s=14)
plt.title("Carat vs Price")
plt.xlabel("carat")
plt.ylabel("price")
save_plot("scatter_carat_price.png")

# box plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="cut", y="price")
plt.title("Price by Cut")
plt.xlabel("Cut")
plt.ylabel("Price")
plt.xticks(rotation=15)
save_plot("box_price_by_cut.png")



# 13) Pearson correlation heatmap

print("\n13) PEARSON CORRELATION")
pearson_corr = df[num_cols].corr()
print(pearson_corr)

# find highest correlated pair
max_val = 0
max_pair = ("", "")
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        a = num_cols[i]
        b = num_cols[j]
        val = abs(pearson_corr.loc[a, b])
        if val > max_val:
            max_val = val
            max_pair = (a, b)
print(f"Highest correlation: {max_pair[0]} vs {max_pair[1]} = {pearson_corr.loc[max_pair[0], max_pair[1]]:.4f}")

plt.figure(figsize=(7, 6))
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Pearson Correlation Heatmap")
save_plot("heatmap_pearson.png")



# 14a) Imputation strategy comparison - Mean vs median for top 2 skewed columns

print("\n14a) IMPUTATION COMPARISON - mean vs median")
top2 = skew_vals.index[:2].tolist()
impute_rows = []
for col in top2:
    impute_rows.append({"column": col, "skew": df[col].skew(),
                        "mean": df[col].mean(), "median": df[col].median(),
                        "nulls_before": int(df[col].isnull().sum())})

impute_df = pd.DataFrame(impute_rows)
print(impute_df.round(4).to_string(index=False))

# use median to fill any remaining nulls in these two columns
for col in top2:
    df[col] = df[col].fillna(df[col].median())

print("Nulls after imputation:")
print(df[top2].isnull().sum())



# 14b) Spearman rank correlation - Spearman vs Pearson

print("\n14b) SPEARMAN VS PEARSON")
spearman_corr = df[num_cols].corr(method="spearman")
print("Pearson:\n", pearson_corr)
print("\nSpearman:\n", spearman_corr)

diff_rows = []
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        a = num_cols[i]
        b = num_cols[j]
        p = pearson_corr.loc[a, b]
        s = spearman_corr.loc[a, b]
        diff_rows.append({"pair": a + " vs " + b, "pearson": p, "spearman": s, "abs_diff": abs(s - p)})

diff_df = pd.DataFrame(diff_rows).sort_values("abs_diff", ascending=False)
print("\nTop 3 pairs with largest difference:")
print(diff_df.head(3))



# 14c) Grouped aggregation

print("\n14c) GROUPED AGGREGATION")
grouped = df.groupby("cut", observed=False)["price"].agg(["mean", "std", "count"])
print(grouped.round(4).to_string())
print("Highest mean group:", grouped["mean"].idxmax())
print("Highest std group:", grouped["std"].idxmax())
print("Mean ratio (max/min):", round(grouped["mean"].max() / grouped["mean"].min(), 4))



# 15) Save cleaned data

print("\n15) SAVE CLEANED DATA")
df.to_csv("cleaned_data.csv", index=False)
print("Saved cleaned_data.csv")

# save tables
null_table.to_csv(OUT + "/null_table.csv")
outlier_df.to_csv(OUT + "/outlier_summary.csv", index=False)
impute_df.to_csv(OUT + "/imputation_comparison.csv", index=False)
pearson_corr.to_csv(OUT + "/pearson_corr.csv")
spearman_corr.to_csv(OUT + "/spearman_corr.csv")
diff_df.to_csv(OUT + "/corr_difference_table.csv", index=False)
grouped.to_csv(OUT + "/grouped_aggregation.csv")

print("Done. Outputs saved to", OUT)
