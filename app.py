from flask import Flask, request, render_template
import pickle
from pathlib import Path

app = Flask(__name__)

# Load model pipeline
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "phishing-detection" / "model"

def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required model artifact: {path}")
    with path.open("rb") as file_handle:
        return pickle.load(file_handle)


model = load_pickle(MODEL_DIR / "phishing_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email_text"]

    prediction = model.predict([email_text])

    result = "Phishing" if prediction[0] == 1 else "Safe"
    return render_template("index.html", prediction_text=f"Email classified as: {result}")

if __name__ == "__main__":
    try:
        from waitress import serve

        serve(app, host="127.0.0.1", port=5000)
    except ImportError:
        app.run(debug=True)
