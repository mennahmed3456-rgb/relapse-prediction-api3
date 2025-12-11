from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# تحميل الموديل
model = joblib.load("final_relapse_model.pkl")

def score_to_class(score):
    if score < 0.25:
        return "Stable"
    elif score < 0.55:
        return "At_Risk"
    else:
        return "Relapsed"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = [
        data["Academic_Performance_Decline"],
        data["Social_Isolation"],
        data["Financial_Issues"],
        data["Physical_Mental_Health_Problems"],
        data["Legal_Consequences"],
        data["Relationship_Strain"],
        data["Risk_Taking_Behavior"],
        data["Withdrawal_Symptoms"],
        data["Denial_and_Resistance_to_Treatment"]
    ]

    features = np.array(features).reshape(1, -1)

    score = float(model.predict(features)[0])
    category = score_to_class(score)

    return jsonify({
        "risk_score": round(score, 4),
        "risk_category": category
    })
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



