import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def generate_synthetic_data(n_samples: int = 15000, random_state: int = 42):
    rng = np.random.default_rng(random_state)

    # Base driving features
    speed = rng.normal(80, 20, n_samples).clip(0, 160)          # km/h
    brake = rng.exponential(20, n_samples).clip(0, 100)         # 0-100
    steering = rng.normal(0, 10, n_samples).clip(-45, 45)       # -45 to 45 degrees
    road = rng.integers(0, 2, n_samples)                        # 0=dry, 1=wet
    traffic = rng.integers(0, 3, n_samples)                     # 0=low,1=medium,2=high
    visibility = rng.normal(80, 15, n_samples).clip(10, 100)    # 0-100
    tyre = rng.integers(0, 2, n_samples)                        # 0=good,1=worn

    # Feature engineering
    speed_brake_ratio = speed / (brake + 1)
    steering_intensity = np.abs(steering)
    visibility_risk = 100 - visibility

    # Heuristic risk scoring to create labels
    score = np.zeros(n_samples)

    # Speed contribution
    score += (speed > 90) * 1.5
    score += (speed > 110) * 2.0
    score += (speed > 130) * 2.0

    # Hard braking
    score += (brake > 50) * 1.0
    score += (brake > 70) * 1.5

    # Aggressive steering
    score += (steering_intensity > 15) * 1.0
    score += (steering_intensity > 25) * 1.5

    # Road condition
    score += (road == 1) * 1.5

    # Traffic
    score += (traffic == 1) * 1.0
    score += (traffic == 2) * 2.0

    # Visibility
    score += (visibility < 60) * 1.0
    score += (visibility < 40) * 1.5

    # Tyre condition
    score += (tyre == 1) * 2.0

    # Map continuous score to discrete classes
    labels = np.empty(n_samples, dtype=object)
    labels[score < 3.0] = "SAFE"
    labels[(score >= 3.0) & (score < 6.5)] = "MEDIUM"
    labels[score >= 6.5] = "HIGH"

    # Stack features in the exact order used by the Flask app
    X = np.column_stack(
        [
            speed,
            brake,
            steering,
            road,
            traffic,
            visibility,
            tyre,
            speed_brake_ratio,
            steering_intensity,
            visibility_risk,
        ]
    )

    return X, labels


def train_and_save_model():
    X, labels = generate_synthetic_data()

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    # Persist artifacts expected by app.py
    joblib.dump(model, "accisense_model.pkl")
    joblib.dump(encoder, "label_encoder.pkl")


if __name__ == "__main__":
    train_and_save_model()

