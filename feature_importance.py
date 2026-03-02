import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ==========================
# LOAD DATASET (for feature names)
# ==========================
df = pd.read_csv("accisense_dataset.csv")

df["speed_brake_ratio"] = df["speed"] / (df["brake"] + 1)
df["steering_intensity"] = abs(df["steering"])
df["visibility_risk"] = 100 - df["visibility"]

X = df.drop("risk", axis=1)


# ==========================
# LOAD TRAINED MODEL
# ==========================
model = joblib.load("accisense_model.pkl")


# ==========================
# GET FEATURE IMPORTANCE
# ==========================
importance = model.feature_importances_
features = X.columns


# ==========================
# SORT FEATURES
# ==========================
imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values("Importance", ascending=False)

print("\nFEATURE IMPORTANCE RANKING\n")
print(imp_df)


# ==========================
# PLOT GRAPH
# ==========================
plt.figure(figsize=(8,5))
plt.barh(imp_df["Feature"], imp_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Accident Risk Feature Importance")
plt.xlabel("Importance Score")
plt.show()
