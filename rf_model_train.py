import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =====================================
# LOAD DATASET
# =====================================
print("Loading dataset...")
df = pd.read_csv("accisense_dataset.csv")
print("Dataset shape:", df.shape)


# =====================================
# FEATURE ENGINEERING (IMPORTANT)
# =====================================
df["speed_brake_ratio"] = df["speed"] / (df["brake"] + 1)
df["steering_intensity"] = abs(df["steering"])
df["visibility_risk"] = 100 - df["visibility"]


# =====================================
# SPLIT FEATURES AND TARGET
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
# TRAIN RANDOM FOREST MODEL
# =====================================
print("Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# =====================================
# EVALUATION
# =====================================
y_pred = model.predict(X_test)

print("\nMODEL ACCURACY:", accuracy_score(y_test, y_pred))
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))


# =====================================
# SAVE MODEL
# =====================================
joblib.dump(model, "accisense_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\nModel saved as accisense_model.pkl")
print("Encoder saved as label_encoder.pkl")
