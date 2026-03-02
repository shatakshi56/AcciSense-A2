from flask import Flask, request, jsonify, render_template
import joblib
import requests
import numpy as np
import warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ================= CONFIG =================
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjU2Mjg3ODYwZDdmZjQ2MDFiMmE2NTBiMjQ5YzlhNzIwIiwiaCI6Im11cm11cjY0In0="

# ================= LOAD MODEL =================
model = joblib.load("accisense_model.pkl")
encoder = joblib.load("label_encoder.pkl")

print("Model loaded successfully!")

# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")

# ================= SAFE ROUTE API =================
@app.route("/get-route", methods=["POST"])
def get_route():
    try:
        data = request.get_json()

        start_lng = float(data["start_lng"])
        start_lat = float(data["start_lat"])
        end_lng = float(data["end_lng"])
        end_lat = float(data["end_lat"])

        body = {
            "coordinates": [
                [start_lng, start_lat],
                [end_lng, end_lat]
            ]
        }

        headers = {
            "Authorization": ORS_API_KEY,
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://api.openrouteservice.org/v2/directions/driving-car",
            json=body,
            headers=headers,
            timeout=10,
            verify=False
        )

        if response.status_code != 200:
            return jsonify({"error": response.text}), 500

        return jsonify(response.json())

    except Exception as e:
        print("ROUTE ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ================= PREDICTION =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        speed      = float(data["speed"])
        brake      = float(data["brake"])
        steering   = float(data["steering"])
        road       = int(data["road"])
        traffic    = int(data["traffic"])
        visibility = float(data["visibility"])
        tyre       = int(data["tyre"])

        speed_brake_ratio  = speed / (brake + 1)
        steering_intensity = abs(steering)
        visibility_risk    = 100 - visibility

        features = np.array([[
            speed, brake, steering, road,
            traffic, visibility, tyre,
            speed_brake_ratio,
            steering_intensity,
            visibility_risk
        ]])

        pred = model.predict(features)
        risk = encoder.inverse_transform(pred)[0]

        # ── Raw risk score (same formula used to label training data) ──
        MAX_RISK = 7.8
        raw_score = (
            speed / 140 +
            brake / 100 +
            abs(steering) / 40 +
            road  * 1.2 +
            traffic * 0.8 +
            (100 - visibility) / 100 +
            tyre  * 1.0
        )

        # ── Piecewise mapping → gauge zone ALWAYS matches risk label ──
        # SAFE   zone: 0  – 32.9%  (raw_score 0  – 2.0)
        # MEDIUM zone: 33 – 65.9%  (raw_score 2.0 – 4.0)
        # HIGH   zone: 66 – 100%   (raw_score 4.0 – 7.8)
        if risk == 'SAFE':
            risk_probability = min(32.9, max(0.0, (raw_score / 2.0) * 33.0))
        elif risk == 'MEDIUM':
            t = min(1.0, max(0.0, (raw_score - 2.0) / 2.0))
            risk_probability = max(33.0, min(65.9, 33.0 + t * 33.0))
        else:  # HIGH
            t = min(1.0, max(0.0, (raw_score - 4.0) / 3.8))
            risk_probability = max(66.0, min(100.0, 66.0 + t * 34.0))
        risk_probability = round(risk_probability, 1)

        # ── Individual contributions (each as % of MAX_RISK) ──
        contributions = [
            {"name": "Speed",        "pct": round((speed / 140) / MAX_RISK * 100, 1),               "raw": f"{speed:.0f} km/h"},
            {"name": "Brake",        "pct": round((brake / 100) / MAX_RISK * 100, 1),               "raw": f"{brake:.0f}%"},
            {"name": "Steering",     "pct": round((abs(steering) / 40) / MAX_RISK * 100, 1),        "raw": f"{steering:.0f}°"},
            {"name": "Road",         "pct": round((road * 1.2) / MAX_RISK * 100, 1),                "raw": "Wet" if road else "Dry"},
            {"name": "Traffic",      "pct": round((traffic * 0.8) / MAX_RISK * 100, 1),             "raw": ["Low", "Medium", "High"][traffic]},
            {"name": "Visibility",   "pct": round(((100 - visibility) / 100) / MAX_RISK * 100, 1), "raw": f"{visibility:.0f}%"},
            {"name": "Tyre",         "pct": round((tyre * 1.0) / MAX_RISK * 100, 1),               "raw": "Worn" if tyre else "Good"},
        ]

        top3 = sorted(contributions, key=lambda x: x["pct"], reverse=True)[:3]

        return jsonify({
            "risk":             risk,
            "speed":            round(speed, 1),
            "brake":            round(brake, 1),
            "steering":         round(steering, 1),
            "road":             road,
            "traffic":          traffic,
            "visibility":       round(visibility, 1),
            "tyre":             tyre,
            "risk_probability": risk_probability,
            "top_conditions":   top3
        })

    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
