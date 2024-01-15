from flask import Flask, send_from_directory, request, jsonify, send_file
from query_language.evaluator import TrajectoryDataset
from initial_models import NUMERICAL_COLUMNS, DISCRETE_EVENT_COLUMNS, COMORBIDITY_FIELDS # todo remove these
from model_training import make_model, load_raw_data, make_modeling_variables, MICROORGANISMS, PRESCRIPTIONS
from model_slice_finding import SliceDiscoveryHelper, SliceEvaluationHelper, describe_slice_change_differences, describe_slice_differences
import slice_finding as sf
import json
import os
import pickle
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
if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)
SLICES_DIR = os.path.join(os.path.dirname(__file__), "slices")
if not os.path.exists(SLICES_DIR): os.mkdir(SLICES_DIR)
TASK_PROGRESS_DIR = os.path.join(os.path.dirname(__file__), "task_progress")

SAMPLE_MODEL_TRAINING_DATA = False

def _background_model_generation(queue):
    finder = None
    dataset = None
    def make_finder():
        return SliceDiscoveryHelper(MODEL_DIR, 
                                        SLICES_DIR, 
                                        min_items_fraction=0.005,
                                        samples_per_model=100,
                                    max_features=4,
                                    scoring_fraction=0.2,
                                    num_candidates=5,
                                    similarity_threshold=0.5)
    while True:
        arg = queue.get()
        if arg == "STOP": return
        if arg[0] == "train_model":
            command, model_name, meta = arg
            try:
                with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                    json.dump({**meta, "training": True, "status": {"state": "loading", "message": "Loading data"}}, file)
                if dataset is None:
                    dataset, (train_patients, val_patients, _) = load_raw_data(sample=SAMPLE_MODEL_TRAINING_DATA)
                with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                    json.dump({**meta, "training": True, "status": {"state": "loading", "message": "Loading variables"}}, file)
                modeling_df = make_modeling_variables(dataset, meta["variables"], meta["timestep_definition"])
                with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                    json.dump({**meta, "training": True, "status": {"state": "loading", "message": "Building model"}}, file)
                make_model(dataset, meta, train_patients, val_patients, save_name=model_name, modeling_df=modeling_df)
                
                if finder is None:
                    finder = make_finder()
                if finder.model_has_slices(model_name):
                    with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                        json.dump({**meta, "training": True, "status": {"state": "loading", "message": "Rescoring slices"}}, file)
                    # Tell the slice finder that the slices need to be rescored for this model
                    finder.rescore_model(model_name)
                with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                    json.dump(meta, file)
            except KeyboardInterrupt:
                with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                    json.dump(meta, file)
                return
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "w") as file:
                    json.dump({**meta, "training": True, "status": {"state": "error", "message": str(e)}}, file)
        elif arg[0] == "find_slices":
            command, model_name, controls = arg
            if finder is None:
                finder = make_finder()
            def filter_single_values(valid_df, outcomes):
                # Exclude any features that have only one value
                single_value_filters = []
                for col_idx, (col, value_pairs) in valid_df.value_names.items():
                    unique_vals = np.unique(valid_df.df[:,col_idx][~pd.isna(outcomes)])
                    if len(unique_vals) == 1:
                        single_value_filters.append(sf.filters.ExcludeFeatureValue(col_idx, unique_vals[0]))
                        
                print("Single value filters:", [(f.feature, f.value) for f in single_value_filters])
                return sf.filters.ExcludeIfAny(single_value_filters)
                
            finder.find_slices(model_name, controls, additional_filter=filter_single_values)
        elif arg[0] == "invalidate_slice_spec":
            command, spec_name = arg
            if finder is None:
                finder = make_finder()
            finder.invalidate_slice_spec(spec_name)

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
    
    if os.path.exists(TASK_PROGRESS_DIR): rmtree(TASK_PROGRESS_DIR)
    os.mkdir(TASK_PROGRESS_DIR)

    # sample_dataset, _ = load_raw_data(sample=True)
    
    sample_dataset, _ = load_raw_data(sample=SAMPLE_MODEL_TRAINING_DATA, cache_dir="data/slicing_variables", val_only=True)
    evaluator = SliceEvaluationHelper(MODEL_DIR, SLICES_DIR)

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
                            "timestep_definition": model_spec["timestep_definition"],
                            "regression": model_spec.get("regression", False),
                            "n_variables": len(model_spec["variables"]),
                            "metrics": json.load(file)
                        }

        return jsonify({ "models": models })

    @app.route("/models/<model_name>/spec")
    def get_model_definition(model_name):
        if not os.path.exists(os.path.join(MODEL_DIR, f"spec_{model_name}.json")):
            return f"Model '{model_name}' does not exist", 400
        
        return send_file(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), mimetype="application/json")

    @app.route("/models/<model_name>/metrics")
    def get_model_metrics(model_name):
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
        queue.put(("train_model", model_name, meta))
        if evaluator is not None:
            # Mark that the model's metrics have changed
            evaluator.rescore_model(model_name, meta["timestep_definition"])
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
            try:
                is_binary = num_unique == 2 and set(np.unique(values).astype(int).tolist()) == set([0, 1])
            except:
                is_binary = False
            if is_binary:
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
        global model_worker
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
        
    @app.route("/slices/<model_name>/start", methods=["POST"])
    def start_slice_finding(model_name):
        if os.path.exists(os.path.join(MODEL_DIR, f"slices_{model_name}.json")):
            with open(os.path.join(MODEL_DIR, f"slices_{model_name}.json"), "r") as file:
                slice_progress = json.load(file)
                if slice_progress.get("searching", False) and slice_progress["status"]["state"] != "error":
                    return "Already finding slices", 400
        else:
            slice_progress = {}
    
        if not os.path.exists(os.path.join(MODEL_DIR, f"spec_{model_name}.json")):
            return "Model does not exist", 404
        
        with open(os.path.join(MODEL_DIR, f"spec_{model_name}.json"), "r") as file:
            spec = json.load(file)

        if spec.get("training", False):
            return "Model is being trained", 400
        
        controls = {}

        try:        
            body = request.json
            if body and "controls" in body:
                controls = body["controls"]
        except:
            pass
        
        if "slice_spec_name" not in controls:
            controls["slice_spec_name"] = "default"
        
        # Start searching
        queue.put(("find_slices", model_name, controls))
        search_status = {"state": "loading", "message": "Starting", "model_name": model_name}
        finder = SliceDiscoveryHelper(MODEL_DIR, SLICES_DIR)
        finder.write_status(True, search_status=search_status)
        return jsonify(finder.get_status())
    
    @app.route("/slices/stop_finding", methods=["POST"])
    def stop_slice_finding():
        global model_worker
        os.kill(model_worker.pid, signal.SIGINT)
        model_worker.join()
        print("Restarting model worker")
        queue = mp.Queue()
        model_worker = mp.Process(target=_background_model_generation, args=(queue,))
        model_worker.start()
      
    @app.route("/slices/status")
    def get_slice_status():
        slice_finder = SliceDiscoveryHelper(MODEL_DIR, SLICES_DIR)
        status = slice_finder.get_status()
        return jsonify(status)

    @app.route("/slices/<model_names>", methods=["POST"])
    def get_slices(model_names):
        model_names = model_names.split(",")
        
        timestep_def = None
        for name in model_names:
            with open(os.path.join(MODEL_DIR, f"spec_{name}.json"), "r") as file:
                spec = json.load(file)
            if spec.get("training", False):
                return "Cannot get slices for model that isn't finished training", 400
            if timestep_def is not None and spec["timestep_definition"] != timestep_def:
                return "Cannot get slices for models with multiple timestep definitions", 400
            timestep_def = spec["timestep_definition"]
            
        weights = evaluator.get_default_evaluation_weights(model_names)
        controls = {}

        try:        
            body = request.json
        except:
            pass
        if body and "controls" in body:
            controls = body["controls"]
            # get weights for additional score functions used by controls
            weights = evaluator.get_default_evaluation_weights(model_names, controls=controls)
        if "slice_spec_name" not in controls:
            controls["slice_spec_name"] = "default"
        if body and "score_weights" in body:
            new_weights = evaluator.weights_for_evaluation(body["score_weights"])
            weights = {k: new_weights.get(k, 0) for k in weights}

        result = evaluator.get_results(timestep_def, controls, model_names)
        if result is None:
            return jsonify({"results": {}, "controls": controls})
        
        rank_list, metrics, ids, df = result
        ranked = rank_list.rank(weights, n_slices=20)
        print(ranked)
        results_json = [
            evaluator.describe_slice(rank_list,
                                     metrics,
                                     ids,
                                     slice_obj,
                                     model_names)
            for slice_obj in ranked
        ]
        base_slice = sf.slices.Slice(sf.slices.SliceFeatureBase())
        response = {
            "state": "complete", 
            "controls": controls,
            "results": {
                "slices": results_json,
                "score_weights": evaluator.weights_for_display(weights),
                "value_names": df.value_names,
                "base_slice": evaluator.describe_slice(rank_list,
                                                       metrics,
                                                       ids,
                                                       base_slice.rescore(rank_list.score_slice(base_slice)), 
                                                       model_names)
            }
        }
        response = sf.utils.convert_to_native_types(response)
        return jsonify(response)
        
    def _score_slice(model_names, timestep_def, slice_spec_name, slice_requests):
        result = evaluator.get_results(timestep_def, {"slice_spec_name": slice_spec_name}, model_names)
        if result is None:
            return {}
        
        rank_list, metrics, ids, df = result
        slices_to_score = {k: rank_list.encode_slice(v) for k, v in slice_requests.items()}
        return sf.utils.convert_to_native_types({
            k: evaluator.describe_slice(rank_list,
                                        metrics, 
                                        ids,
                                        slice_obj.rescore(rank_list.score_slice(slice_obj)),
                                        model_names)
            for k, slice_obj in slices_to_score.items()
        })
        
    @app.route("/slices/<model_names>/score", methods=["POST"])
    def score_slice(model_names):
        model_names = model_names.split(",")
        
        timestep_def = None
        for name in model_names:
            with open(os.path.join(MODEL_DIR, f"spec_{name}.json"), "r") as file:
                spec = json.load(file)
            if timestep_def is not None and spec["timestep_definition"] != timestep_def:
                return "Cannot get slices for models with multiple timestep definitions", 400
            timestep_def = spec["timestep_definition"]
            
        body = request.json
        if "sliceRequests" not in body:
            return "'sliceRequests' key required", 400
        
        slice_spec_name = body.get("sliceSpec", "default")
        
        if len(body["sliceRequests"]) == 0:
            return jsonify({"sliceRequestResults": {}})
        
        results_json = _score_slice(model_names, timestep_def, slice_spec_name, body["sliceRequests"])
        return jsonify({ "sliceRequestResults": results_json })
    
    @app.route("/slices/score", methods=["POST"])
    def score_slice_all_models():
        timestep_defs = {}
        for path in os.listdir(MODEL_DIR):
            if path.startswith("spec_"):
                model_id = re.search(r'^spec_(.*)\.json', path).group(1)
                with open(os.path.join(MODEL_DIR, path), "r") as file:
                    model_spec = json.load(file)
                if not model_spec.get("training", False):
                    timestep_defs.setdefault(model_spec["timestep_definition"], []).append(model_id)
        
        body = request.json
        if "sliceRequests" not in body:
            return "'sliceRequests' key required", 400
        
        if len(body["sliceRequests"]) == 0:
            return jsonify({"sliceRequestResults": {}})
        
        slice_spec_name = body.get("sliceSpec", "default")
        
        results = {}
        for timestep_def, model_names in timestep_defs.items():
            ts_results = _score_slice(model_names, timestep_def, slice_spec_name, body["sliceRequests"])
            for slice_key in ts_results:
                if slice_key not in results:
                    results[slice_key] = ts_results[slice_key]
                else:
                    # Merge the slice definitions
                    results[slice_key]["score_values"].update(ts_results[slice_key]["score_values"])
                    results[slice_key]["metrics"].update(ts_results[slice_key]["metrics"])
        return jsonify({ "sliceRequestResults": results })
    
    @app.route("/slices/<model_names>/compare", methods=["POST"])
    def get_slice_comparisons(model_names):
        model_names = model_names.split(",")
        
        timestep_def = None
        for name in model_names:
            with open(os.path.join(MODEL_DIR, f"spec_{name}.json"), "r") as file:
                spec = json.load(file)
            if timestep_def is not None and spec["timestep_definition"] != timestep_def:
                return "Cannot get slices for models with multiple timestep definitions", 400
            timestep_def = spec["timestep_definition"]
            
        body = request.json
        if "slice" not in body:
            return "'slice' body argument required", 400
        
        discrete_df, eval_mask, ids, slice_filter = evaluator.get_slicing_data(body.get("sliceSpec", "default"),
                                                                 timestep_def,
                                                                 evaluation=True)
        slice_to_eval = discrete_df.encode_slice(body["slice"])
        
        offset = body.get("offset", None)
        
        metrics = evaluator.get_eval_metrics(model_names, eval_mask)
        valid_mask = np.all(np.vstack(list(metrics.values())), axis=0)
        if offset is not None:
            try:
                if int(offset) != offset:
                    return "Offset must be an integer", 400
            except ValueError:
                return "Offset must be an integer", 400
            
            # Return a comparison of the slice at the given number of steps
            # offset from the current time compared to the current time
            comparison = describe_slice_change_differences(
                discrete_df,
                ids,
                offset,
                slice_to_eval,
                slice_filter=slice_filter,
                valid_mask=valid_mask
            )
            return jsonify(sf.utils.convert_to_native_types(comparison))
        else:
            differences = describe_slice_differences(
                discrete_df,
                slice_to_eval,
                slice_filter=slice_filter,
                valid_mask=valid_mask
            )
            return jsonify(sf.utils.convert_to_native_types(differences))
        
        
    @app.route("/slices/specs", methods=["GET"])
    def get_slice_specs():
        slice_spec_dir = os.path.join(SLICES_DIR, "specifications")
        results = {}
        for path in os.listdir(slice_spec_dir):
            if not path.endswith(".json"): continue
            spec_name = os.path.splitext(path)[0]
            with open(os.path.join(slice_spec_dir, path), "r") as file:
                results[spec_name] = json.load(file)
        return jsonify(results)
    
    @app.route("/slices/specs/<spec_name>", methods=["POST"])
    def edit_slice_spec(spec_name):
        slice_spec_dir = os.path.join(SLICES_DIR, "specifications")
        body = request.json
        if "variables" not in body or "slice_filter" not in body:
            return "Slice spec must contain 'variables' and 'slice_filter' keys", 400
        
        spec_path = os.path.join(slice_spec_dir, f"{spec_name}.json")
        if os.path.exists(spec_path):
            print(f"Invalidating existing slices with spec {spec_name}")
            queue.put(("invalidate_slice_spec", spec_name))
            evaluator.invalidate_slice_spec(spec_name)
            
        with open(spec_path, "w") as file:
            json.dump(body, file)
            
        return "Success"
            
        
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