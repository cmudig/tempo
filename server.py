# Change to import `send_from_directory`
from flask import Flask, send_from_directory, request, jsonify

app = Flask(__name__)


@app.route('/models')
def get_models():
    return jsonify({
        "models": [
            {
                "target_name": "Invasive Ventilation in 24h",
                "n_patients": 12409,
                "n_timesteps": 2049388,
                "type": "classification",
                "metrics": {
                    "roc_auc": 0.817,
                    "sensitivity": 0.91,
                    "specificity": 0.68
                }
            },
            {
                "target_name": "Antibiotics in 24h",
                "n_patients": 6827,
                "n_timesteps": 182735,
                "type": "classification",
                "likely_trivial": True,
                "metrics": {
                    "roc_auc": 0.981,
                    "sensitivity": 0.97,
                    "specificity": 0.95
                }
            },
            {
                "target_name": "Vasopressors in 12h",
                "n_patients": 9189,
                "n_timesteps": 980138,
                "type": "classification",
                "metrics": {
                    "roc_auc": 0.951,
                    "sensitivity": 0.92,
                    "specificity": 0.89
                }
            },
            {
                "target_name": "mAP in 24h",
                "n_patients": 14857,
                "n_timesteps": 3492809,
                "type": "regression",
                "metrics": {
                    "r2_score": 0.566,
                }
            }
        ]
    })

# Path for our main Svelte page
@app.route("/")
def client():
    return send_from_directory('client/dist', 'index.html')

# Route to add static files (CSS and JS)
@app.route("/<path:path>")
def base(path):
    return send_from_directory('client/dist', path)

if __name__ == '__main__':
    app.run(debug=True)