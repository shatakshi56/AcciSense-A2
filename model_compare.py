import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =====================================
# LOAD DATASET
# =====================================
print("Loading dataset...")
df = pd.read_csv("accisense_dataset.csv")


# =====================================
# FEATURE ENGINEERING
# =====================================
df["speed_brake_ratio"] = df["speed"] / (df["brake"] + 1)
df["steering_intensity"] = abs(df["steering"])
df["visibility_risk"] = 100 - df["visibility"]


# =====================================
# FEATURES & TARGET
# =====================================
X = df.drop("risk", axis=1)
y = df["risk"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)


# =====================================
# TRAIN TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================================
# RANDOM FOREST TRAIN
# =====================================
print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# =====================================
# XGBOOST TRAIN
# =====================================
print("\nTraining XGBoost...")

xgb = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)


# =====================================
# PERFORMANCE
# =====================================
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

rf_f1 = f1_score(y_test, rf_pred, average="weighted")
xgb_f1 = f1_score(y_test, xgb_pred, average="weighted")

print("\n========== MODEL COMPARISON ==========")
print("Random Forest Accuracy:", rf_acc)
print("XGBoost Accuracy:", xgb_acc)
print("Random Forest F1:", rf_f1)
print("XGBoost F1:", xgb_f1)


print("\n===== RANDOM FOREST REPORT =====")
print(classification_report(y_test, rf_pred))

print("\n===== XGBOOST REPORT =====")
print(classification_report(y_test, xgb_pred))


print("\nRandom Forest Confusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nXGBoost Confusion Matrix")
print(confusion_matrix(y_test, xgb_pred))


# =====================================
# BEST MODEL
# =====================================
best = "Random Forest" if rf_f1 > xgb_f1 else "XGBoost"
print("\nBEST MODEL:", best)
