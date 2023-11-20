from flask import Flask, send_from_directory, request, jsonify, send_file
from flask_apscheduler import APScheduler
from query_language.evaluator import TrajectoryDataset
from model_training import make_model, load_raw_data, make_modeling_variables
import json
import os
from shutil import rmtree
from functools import partial
import re

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TASK_PROGRESS_DIR = os.path.join(os.path.dirname(__file__), "task_progress")
if os.path.exists(TASK_PROGRESS_DIR): rmtree(TASK_PROGRESS_DIR)
os.mkdir(TASK_PROGRESS_DIR)

app = Flask(__name__)

# initialize scheduler
scheduler = APScheduler()
scheduler.api_enabled = True
scheduler.init_app(app)
scheduler.start()

@app.route('/models', methods=["GET"])
def get_models():
    return jsonify({
        "models": {
            "vasopressor_8h": {
                "target_name": "Invasive Ventilation in 24h",
                "type": "classification",
                "metrics": {
                    "roc_auc": 0.817,
                    "sensitivity": 0.91,
                    "specificity": 0.68
                }
            },
            "antimicrobial_8h": {
                "target_name": "Antibiotics in 24h",
                "type": "classification",
                "likely_trivial": True,
                "metrics": {
                    "roc_auc": 0.981,
                    "sensitivity": 0.97,
                    "specificity": 0.95
                }
            },
            "ventilation_8h": {
                "target_name": "Vasopressors in 12h",
                "type": "classification",
                "metrics": {
                    "roc_auc": 0.951,
                    "sensitivity": 0.92,
                    "specificity": 0.89
                }
            }
        }
    })

@app.route("/models/<model_name>/spec")
def get_model_definition(model_name):
    if not re.match("^[A-Za-z0-9_-]+$", model_name):
        return f"Invalid model name {model_name}", 400
    if not os.path.exists(os.path.join(MODEL_DIR, f"spec_{model_name}.json")):
        return f"Model '{model_name}' does not exist", 400
    
    return send_file(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), mimetype="application/json")

@app.route("/models/<model_name>/metrics")
def get_model_metrics(model_name):
    if not re.match("^[A-Za-z0-9_-]+$", model_name):
        return f"Invalid model name {model_name}", 400
    if not os.path.exists(os.path.join(MODEL_DIR, f"spec_{model_name}.json")):
        return f"Model '{model_name}' does not exist", 400
    
    return send_file(os.path.join(MODEL_DIR, f"metrics_{model_name}.json"), mimetype="application/json")

def _background_model_generation(model_name, meta):
    try:
        with open(os.path.join(TASK_PROGRESS_DIR, "model_gen.json"), "w") as file:
            json.dump({"model_name": model_name, "status": "loading", "message": "Loading data"}, file)
        dataset, (train_patients, val_patients, _) = load_raw_data()
        with open(os.path.join(TASK_PROGRESS_DIR, "model_gen.json"), "w") as file:
            json.dump({"model_name": model_name, "status": "loading", "message": "Loading variables"}, file)
        modeling_df = make_modeling_variables(dataset, meta["variables"], meta["timestep_definition"])
        with open(os.path.join(TASK_PROGRESS_DIR, "model_gen.json"), "w") as file:
            json.dump({"model_name": model_name, "status": "loading", "message": "Building model"}, file)
        make_model(dataset, meta, train_patients, val_patients, save_name=model_name, modeling_df=modeling_df)
        with open(os.path.join(TASK_PROGRESS_DIR, "model_gen.json"), "w") as file:
            json.dump({"model_name": model_name, "status": "complete", "message": "Complete"}, file)
    except Exception as e:
        print("Error loading model:", e)
        with open(os.path.join(TASK_PROGRESS_DIR, "model_gen.json"), "w") as file:
            json.dump({"model_name": model_name, "status": "error", "message": str(e)}, file)
    
def _get_model_training_status():
    if os.path.exists(os.path.join(TASK_PROGRESS_DIR, "model_gen.json")):
        with open(os.path.join(TASK_PROGRESS_DIR, "model_gen.json"), "r") as file:
            status = json.load(file)
            return status
    return {"status": "none", "message": "No model being trained"}
            
@app.route("/models", methods=["POST"])
def generate_model():
    if _get_model_training_status()["status"] == "loading":
        return "Another model is currently being trained; please try again later.", 400    
    
    body = request.json
    if "name" not in body:
        return "Model 'name' key required", 400
    model_name = body["name"]
    if "meta" not in body:
        return "Model 'meta' key required", 400
    meta = body["meta"]
    
    with open(os.path.join(TASK_PROGRESS_DIR, "model_gen.json"), "w") as file:
        json.dump({"model_name": model_name, "status": "loading", "message": "Starting"}, file)
    scheduler.add_job("model_gen", partial(_background_model_generation, model_name, meta))
    return f"Started training model '{model_name}'", 200
    
@app.route("/training_status")
def model_training_status():
    return jsonify(_get_model_training_status())
    
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