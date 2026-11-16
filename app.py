
from flask import Flask, jsonify
from flask_cors import CORS
from disease_prediction import disease_bp

app = Flask(__name__)

CORS(app)

app.register_blueprint(
    disease_bp,
    url_prefix="/api/disease"
)

@app.route("/")
def home():
    return jsonify({
        "message": "Pesticide API is running successfully"
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
