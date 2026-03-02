import pandas as pd

# load dataset
df = pd.read_csv("accisense_dataset.csv")

print("Dataset shape:", df.shape)


# ===============================
# IQR OUTLIER DETECTION
# ===============================
def detect_outliers(column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(f"\n{column}")
    print("Lower bound:", lower)
    print("Upper bound:", upper)
    print("Outliers count:", len(outliers))


# numeric columns
numeric_cols = ["speed","brake","steering","visibility"]

for col in numeric_cols:
    detect_outliers(col)
