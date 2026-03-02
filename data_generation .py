import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)


# =====================================
# SENSOR DATA GENERATOR
# =====================================
def generate_sensor_data():

    speed = np.clip(random.gauss(70, 20), 20, 140)
    brake = np.clip(random.gauss(40, 25), 0, 100)
    steering = np.clip(random.gauss(0, 15), -40, 40)

    road = random.choice([0, 1])        # 0 dry, 1 wet
    traffic = random.choice([0, 1, 2])  # low medium high
    visibility = np.clip(random.gauss(70, 25), 5, 100)
    tyre = random.choice([0, 1])        # good worn

    # risk score logic
    risk_score = (
        speed / 140 +
        brake / 100 +
        abs(steering) / 40 +
        road * 1.2 +
        traffic * 0.8 +
        (100 - visibility) / 100 +
        tyre * 1.0
    )

    if risk_score > 4:
        risk = "HIGH"
    elif risk_score > 2:
        risk = "MEDIUM"
    else:
        risk = "SAFE"

    return [
        speed,
        brake,
        steering,
        road,
        traffic,
        visibility,
        tyre,
        risk
    ]


# =====================================
# CREATE DATASET
# =====================================
print("Generating synthetic driving dataset...")

data = [generate_sensor_data() for _ in range(15000)]

df = pd.DataFrame(data, columns=[
    "speed",
    "brake",
    "steering",
    "road",
    "traffic",
    "visibility",
    "tyre",
    "risk"
])

# save dataset
df.to_csv("accisense_dataset.csv", index=False)

print("Dataset saved as accisense_dataset.csv")
print("Shape:", df.shape)
print(df.head())
