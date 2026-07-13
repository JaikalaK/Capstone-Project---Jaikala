# Part 1 - Data Cleaning and EDA

## What this is
This is part 1 of the capstone project. I did data cleaning and exploratory analysis on the diamonds dataset. The goal was to understand the data before any modeling.

## Dataset
I used the publicly available Diamonds dataset from seaborn's data repo:
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv

It has 53940 rows and 10 columns. I picked this because it has a clear numeric target (price), a few categorical columns (cut, color, clarity), and it's free to download without any sign-up.

## How to run
```bash
pip install -r requirements.txt
python part1_eda.py
```
Add `--show-plots` at the end if you want the charts to open on screen. By default they just get saved to `outputs/part1/figures/`.

## What the script does

**Step 1 - Load data**
Just reads the CSV and prints the first few rows, dtypes, and shape. I loaded `table` as object type on purpose to demonstrate the dtype correction in step 4.

**Step 2 - Null values**
Checked all columns for missing values. None of the columns had nulls in this dataset. For any column below 20% null rate, I filled with the column median.

I chose median over mean because the price and size columns are right-skewed. When data is skewed, the mean gets pulled toward the extreme values, so the median is a better middle-ground estimate.

**Step 3 - Duplicates**
Found 146 duplicate rows and removed them. After removing duplicates the null percentages didn't change, which is what you'd expect.

**Step 4 - Fix data types**
`table` column came in as object (string) even though it's numeric. I used `pd.to_numeric(..., errors='coerce')` to fix it. I also converted `cut`, `color`, and `clarity` to category dtype which saved about 12 million bytes of memory.

**Step 5 - Skewness**
The column `y` (diamond width in mm) had the highest skewness value of about 2.45. This is a positive skew meaning there are a few very wide diamonds pulling the distribution to the right. Using the mean to fill missing values here would overestimate what a typical diamond looks like.

**Step 6 - IQR outlier check (price and carat)**

For `price`, the IQR method gave a lower bound of -5612.63 and an upper bound of 11890.38. Based on those limits, 3523 rows were flagged as outliers.

For `carat`, the lower bound came out to -0.56 and the upper bound came out to 2.00. Using those limits, 1873 rows were flagged as outliers.

I didn't remove these outliers. They are likely real high-value diamonds, not bad data. I'll revisit this in part 2 if needed.

**Step 7-12 - Plots**
Made 5 different chart types as required:
- Line plot: price across row index
- Bar chart: mean price per cut category
- Histogram: the most skewed column (y)
- Scatter: carat vs price — strong positive relationship, clearly not random
- Box plot: price split by cut — different medians per category, lots of spread in premium cuts

**Step 13 - Pearson heatmap**
The strongest correlation is carat vs x (0.975). This makes sense because carat is weight and x is length, both relate to the physical size of the diamond. This doesn't necessarily mean one causes the other — they're both driven by how big the stone is.

**Step 14a - Imputation strategy comparison - Mean vs Median for top 2 skewed columns**

The two most skewed columns were `y` and `price`. For `y`, the skewness was about 2.45, the mean was 5.7347, and the median was 5.71. The mean and median are close, but the mean is still a little higher because of the right tail.

For `price`, the skewness was about 1.62, the mean was 3933.07, and the median was 2401.0. This is a much bigger gap and it shows clearly how a few expensive diamonds pull the mean upward.

For `price` the mean is much higher than the median (3933 vs 2401). This shows how skewed distributions make mean a bad imputation choice. I used median for both.

**Step 14b - Spearman rank correlation - Spearman vs Pearson**
Computed both correlation matrices and checked where they differ. The three pairs with largest difference are price vs y, price vs z, and price vs x. Spearman is higher than Pearson for all three, meaning these relationships are monotonic but not strictly linear. For feature selection in part 2 I'll use Spearman as the main guide.

**Step 14c - Grouped stats by cut**

When I grouped price by `cut`, the `Premium` group had the highest mean price at about 4583.5. It also had the highest standard deviation at about 4348.1, which means prices inside that group are spread out a lot.

`Fair` had a mean around 4341.9 with 1598 rows, `Good` had a mean around 3919.1 with 4891 rows, `Ideal` had a mean around 3462.7 with 21488 rows, and `Very Good` had a mean around 3981.0 with 12069 rows.

Premium has the highest mean and also the highest standard deviation. The ratio of highest to lowest mean is 1.32, which suggests cut does carry some signal but there's a lot of overlap between groups. High std means the feature alone isn't enough to predict price precisely.

**Step 15 - Save**
Saved the cleaned file as `cleaned_data.csv` which gets used in parts 2, 3, and 4.


