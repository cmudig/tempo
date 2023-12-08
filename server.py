from flask import Flask, send_from_directory, request, jsonify, send_file
from query_language.evaluator import TrajectoryDataset
from model_training import make_model, load_raw_data, make_modeling_variables
import json
import os
import signal
import pandas as pd
import numpy as np
from shutil import rmtree
from functools import partial
import re
import time
import multiprocessing as mp
import atexit

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TASK_PROGRESS_DIR = os.path.join(os.path.dirname(__file__), "task_progress")
if os.path.exists(TASK_PROGRESS_DIR): rmtree(TASK_PROGRESS_DIR)
os.mkdir(TASK_PROGRESS_DIR)

SAMPLE_MODEL_TRAINING_DATA = False

def _background_model_generation(queue):
    while True:
        arg = queue.get()
        if arg == "STOP": return
        model_name, meta = arg
        try:
            with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                json.dump({**meta, "training": True, "status": {"state": "loading", "message": "Loading data"}}, file)
            dataset, (train_patients, val_patients, _) = load_raw_data(sample=SAMPLE_MODEL_TRAINING_DATA)
            with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                json.dump({**meta, "training": True, "status": {"state": "loading", "message": "Loading variables"}}, file)
            modeling_df = make_modeling_variables(dataset, meta["variables"], meta["timestep_definition"])
            with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                json.dump({**meta, "training": True, "status": {"state": "loading", "message": "Building model"}}, file)
            make_model(dataset, meta, train_patients, val_patients, save_name=model_name, modeling_df=modeling_df)
        except KeyboardInterrupt:
            with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                json.dump(meta, file)
            return
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                json.dump({**meta, "training": True, "status": {"state": "error", "message": str(e)}}, file)
    
def _get_model_training_status(model_name):
    if os.path.exists(os.path.join(MODEL_DIR, f"spec_{model_name}.json")):
        with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "r") as file:
            spec = json.load(file)
            if spec.get("training", False):
                return spec["status"]
    return {"state": "none", "message": "No model being trained"}
            
def _is_model_training():
    for path in os.listdir(MODEL_DIR):
        if path.startswith("spec_"):
            model_id = re.search(r'^spec_(.*)\.json', path).group(1)
            with open(os.path.join(MODEL_DIR, path), "r") as file:
                model_spec = json.load(file)
            if model_spec.get("training", False):
                return True
    return False

if __name__ == '__main__':
    app = Flask(__name__)
    
    queue = mp.Queue()
    model_worker = mp.Process(target=_background_model_generation, args=(queue,))
    model_worker.start()

    sample_dataset, _ = load_raw_data(sample=True)

    @app.route('/models', methods=["GET"])
    def get_models():
        models = {}
        for path in os.listdir(MODEL_DIR):
            if path.startswith("spec_"):
                model_id = re.search(r'^spec_(.*)\.json', path).group(1)
                with open(os.path.join(MODEL_DIR, path), "r") as file:
                    model_spec = json.load(file)
                if model_spec.get("training", False):
                    models[model_id] = model_spec
                else:
                    with open(os.path.join(MODEL_DIR, f"metrics_{model_id}.json"), "r") as file:
                        models[model_id] = {
                            "outcome": model_spec["outcome"],
                            "regression": model_spec.get("regression", False),
                            "n_variables": len(model_spec["variables"]),
                            "metrics": json.load(file)
                        }

        return jsonify({ "models": models })

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
        
        if not os.path.exists(os.path.join(MODEL_DIR, f"metrics_{model_name}.json")):
            return f"No metrics available", 400
        with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "r") as file:
            model_spec = json.load(file)
            if model_spec.get("training", False):
                return f"No metrics available", 400
        return send_file(os.path.join(MODEL_DIR, f"metrics_{model_name}.json"), mimetype="application/json")

    @app.route("/models", methods=["POST"])
    def generate_model():
        body = request.json
        if "name" not in body:
            return "Model 'name' key required", 400
        model_name = body["name"]
        if "meta" not in body:
            return "Model 'meta' key required", 400
        meta = body["meta"]
        
        if _is_model_training():
            with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                json.dump({**meta, "training": True, "status": { 
                    "state": "waiting", 
                    "message": "Waiting for other training jobs to complete"
                }}, file)
        else:
            with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                json.dump({**meta, "training": True, "status": { 
                    "state": "loading", 
                    "message": "Starting"
                }}, file)
        queue.put((model_name, meta))
        return f"Started training model '{model_name}'", 200
        
    @app.route("/data/query")
    def query_dataset():
        args = request.args
        if "q" not in args: return "Query endpoint must have a 'q' query argument", 400
        try:
            result = sample_dataset.query(args.get("q"), use_cache=False)
            summary = {
                "n_values": len(result.get_values()),
                "n_trajectories": len(set(result.get_ids().values.tolist())),
                "query": args.get("q")
            }
            values = result.get_values()
            num_unique = len(np.unique(values))
            if num_unique == 2 and set(np.unique(values).astype(int).tolist()) == set([0, 1]):
                summary["type"] = "binary"
                summary["rate"] = values.mean().astype(float)
            elif pd.api.types.is_object_dtype(values.dtype) or num_unique <= 10:
                summary["type"] = "categorical"
                summary["counts"] = {str(k): int(v) for k, v in zip(*np.unique(values, return_counts=True))}
            else:
                summary["type"] = "continuous"
                summary["mean"] = np.mean(values.astype(float))
                summary["std"] = np.std(values.astype(float))
                
                min_val = values.min()
                max_val = values.max()
                data_range = max_val - min_val
                bin_scale = np.floor(np.log10(data_range))
                if data_range / (10 ** bin_scale) < 2.5:
                    bin_scale -= 1 # Make sure there aren't only 2-3 bins
                upper_tol = 2 if (np.ceil(max_val / (10 ** bin_scale))) * (10 ** bin_scale) == max_val else 1
                hist_bins = np.arange(np.floor(min_val / (10 ** bin_scale)) * (10 ** bin_scale),
                                        (np.ceil(max_val / (10 ** bin_scale)) + upper_tol) * (10 ** bin_scale),
                                        10 ** bin_scale)
                
                summary["hist"] = {"counts": np.histogram(values, bins=hist_bins)[0].astype(int).tolist(), 
                                "bins": hist_bins.astype(float).tolist()}
            return jsonify({"summary": summary})
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return jsonify({"error": str(e)})
        
        
    @app.route("/models/status/<model_name>")
    def model_training_status(model_name):
        return jsonify(_get_model_training_status(model_name))
        
    @app.route("/models/stop_training/<model_name>", methods=["POST"])
    def stop_model_training(model_name):
        state = _get_model_training_status(model_name)["state"]
        if state == "loading":
            os.kill(model_worker.pid, signal.SIGINT)
            model_worker.join()
            print("Restarting model worker")
            queue = mp.Queue()
            model_worker = mp.Process(target=_background_model_generation, args=(queue,))
            model_worker.start()
        elif state in ("waiting", "error"):
            with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "r") as file:
                meta = json.load(file)
            with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                json.dump({k: v for k, v in meta.items() if k not in ("training", "status")}, file)
        
    # Path for our main Svelte page
    @app.route("/")
    def client():
        return send_from_directory('client/dist', 'index.html')

    # Route to add static files (CSS and JS)
    @app.route("/<path:path>")
    def base(path):
        return send_from_directory('client/dist', path)

    def close_running_threads():
        queue.put('STOP')
        model_worker.join()
        print("Shut down model worker")
        
    atexit.register(close_running_threads)
    
    app.run(debug=True, port=4999)