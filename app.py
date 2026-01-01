from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and transformer
with open("poly_transform.pkl", "rb") as f:
    poly = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def dosage_threshold(dosage):
    if dosage < 30:
        return "UNDERDOSE – Ineffective"
    elif dosage > 120:
        return "OVERDOSE – Side Effects Risk"
    else:
        return "SAFE – Optimal Range"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    dosage = float(data["dosage"])
    dosage_poly = poly.transform(np.array([[dosage]]))
    recovery_rate = model.predict(dosage_poly)[0]
    threshold = dosage_threshold(dosage)

    # Generate curve for visualization
    dosage_range = np.linspace(0, 200, 100).reshape(-1, 1)
    dosage_range_poly = poly.transform(dosage_range)
    recovery_curve = model.predict(dosage_range_poly)

    return jsonify({
        "recovery_rate": recovery_rate,
        "threshold": threshold,
        "curve_x": dosage_range.flatten().tolist(),
        "curve_y": recovery_curve.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)
