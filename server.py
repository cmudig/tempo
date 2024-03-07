from flask import Flask, send_from_directory, request, jsonify, send_file
from dataset_manager import DatasetManager
from query_language.data_types import QUERY_RESULT_TYPENAMES
from model_training import make_modeling_variables
from model_slice_finding import describe_slice_change_differences, describe_slice_differences
from utils import make_query_result_summary
from threading import Lock
from io import BytesIO
import zipfile
import time
import slice_finding as sf
import datetime
import json
import lark
import os
import sys
import signal
import numpy as np
import re
import multiprocessing as mp
import atexit
import argparse
from werkzeug.middleware.profiler import ProfilerMiddleware

SAMPLE_MODEL_TRAINING_DATA = False

def _background_model_generation(base_path, queue, single_thread=False):
    finder = None
    dataset = None
    dataset_manager = DatasetManager(base_path)
    trainer = dataset_manager.make_trainer()
    def make_finder():
        f = dataset_manager.make_slice_discovery_helper()
        f.n_workers = 1 if single_thread else None
        return f
    while True:
        arg = queue.get()
        if arg == "STOP": return
        if arg[0] == "train_model":
            command, model_name, meta = arg
            try:
                with open(dataset_manager.model_spec_path(model_name), "w") as file:
                    json.dump({**meta, "training": True, "status": {"state": "loading", "message": "Loading data"}}, file)
                if dataset is None:
                    dataset, (train_ids, val_ids, test_ids) = dataset_manager.load_data(sample=SAMPLE_MODEL_TRAINING_DATA)
                with open(dataset_manager.model_spec_path(model_name), "w") as file:
                    json.dump({**meta, "training": True, "status": {"state": "loading", "message": "Loading variables"}}, file)
                modeling_df = make_modeling_variables(dataset, meta["variables"], meta["timestep_definition"])
                with open(dataset_manager.model_spec_path(model_name), "w") as file:
                    json.dump({**meta, "training": True, "status": {"state": "loading", "message": "Building model"}}, file)
                trainer.make_model(dataset, meta, train_ids, val_ids, test_ids, save_name=model_name, modeling_df=modeling_df)
                
                if finder is None:
                    finder = make_finder()
                if finder.model_has_slices(model_name):
                    with open(dataset_manager.model_spec_path(model_name), "w") as file:
                        json.dump({**meta, "training": True, "status": {"state": "loading", "message": "Rescoring slices"}}, file)
                    # Tell the slice finder that the slices need to be rescored for this model
                    finder.rescore_model(model_name)
                with open(dataset_manager.model_spec_path(model_name), "w") as file:
                    json.dump(meta, file)
            except KeyboardInterrupt:
                with open(dataset_manager.model_spec_path(model_name), "w") as file:
                    json.dump(meta, file)
                return
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                with open(dataset_manager.model_spec_path(model_name), "w") as file:
                    json.dump({**meta, "training": True, "status": {"state": "error", "message": str(e)}}, file)
        elif arg[0] == "find_slices":
            command, model_name, controls = arg
            if finder is None:
                finder = make_finder()
            finder.find_slices(model_name, controls)
        elif arg[0] == "invalidate_slice_spec":
            command, spec_name = arg
            if finder is None:
                finder = make_finder()
            finder.invalidate_slice_spec(spec_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='TimeseriesModelExplorer',
                description='Start a Flask server to create, interpret, and compare time-series prediction model formulations.')

    parser.add_argument('base_path', type=str)
    parser.add_argument('--port', type=int, default=4999, help='Port to run the server on')
    parser.add_argument('--public', action='store_true', default=False, help='Open the server to public network traffic')
    parser.add_argument('--profile', action='store_true', default=False, help='Print cProfile performance logs for each API call')
    parser.add_argument('--single-thread', action='store_true', default=False, help='Disable multithreading for slice finding')
    
    args = parser.parse_args()

    app = Flask(__name__)
    
    base_path = args.base_path
    dataset_manager = DatasetManager(base_path)
    MODEL_DIR = dataset_manager.model_dir
    queue = mp.Queue()
    model_worker = mp.Process(target=_background_model_generation, args=(base_path, queue), kwargs={"single_thread": args.single_thread})
    model_worker.start()
    model_worker_comm_lock = Lock() # make sure we are not issuing simultaneous commands to the model worker
    
    sample_dataset, _ = dataset_manager.load_data(sample=SAMPLE_MODEL_TRAINING_DATA, 
                                                  cache_dir=os.path.join(dataset_manager.base_path, "data/sample" if SAMPLE_MODEL_TRAINING_DATA else "slices/slicing_variables"), 
                                                  split='test')
    evaluator = dataset_manager.make_slice_evaluation_helper()
    evaluator_lock = Lock() # make sure we are not requesting slices simultaneously from different requests

    def _get_model_training_status(model_name):
        if os.path.exists(dataset_manager.model_spec_path(model_name)):
            with open(dataset_manager.model_spec_path(model_name), "r") as file:
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

    @app.route('/models', methods=["GET"])
    def get_models():
        models = {}
        for path in os.listdir(dataset_manager.model_dir):
            if path.startswith("spec_"):
                model_id = re.search(r'^spec_(.*)\.json', path).group(1)
                with open(os.path.join(MODEL_DIR, path), "r") as file:
                    model_spec = json.load(file)
                if model_spec.get("training", False) or not os.path.exists(dataset_manager.model_metrics_path(model_id)):
                    models[model_id] = model_spec
                else:
                    with open(dataset_manager.model_metrics_path(model_id), "r") as file:
                        models[model_id] = {
                            "outcome": model_spec["outcome"],
                            "timestep_definition": model_spec["timestep_definition"],
                            "model_type": model_spec.get("model_type", "binary_classification"),
                            "variables": model_spec["variables"],
                            "cohort": model_spec.get("cohort", ""),
                            "metrics": json.load(file),
                            **({"output_values": model_spec["output_values"]} if "output_values" in model_spec else {})
                        }

        return jsonify({ "models": models })

    @app.route("/models/<model_name>/spec")
    def get_model_definition(model_name):
        if not os.path.exists(dataset_manager.model_spec_path(model_name)):
            return f"Model '{model_name}' does not exist", 400
        
        return send_file(dataset_manager.model_spec_path(model_name), mimetype="application/json")

    @app.post("/models/new/<reference_name>")
    def make_new_model_spec(reference_name):
        if reference_name == "default":
            base_spec = dataset_manager.get_default_model_spec()
            base_name = "Untitled"
        else:
            if not os.path.exists(dataset_manager.model_spec_path(reference_name)):
                return f"Model '{reference_name}' does not exist", 400
            with open(dataset_manager.model_spec_path(reference_name), "r") as file:
                base_spec = json.load(file)
            base_name = reference_name
            
        increment_index = None
        final_name = base_name
        while os.path.exists(dataset_manager.model_spec_path(final_name)):
            if increment_index is None:
                increment_index = 2
            else:
                increment_index += 1
            final_name = f"{base_name} {increment_index}"
            
        with open(dataset_manager.model_spec_path(final_name), "w") as file:
            json.dump({**base_spec, "training": False}, file)
        return jsonify({"name": final_name, "spec": base_spec})
    
    @app.delete("/models/<model_name>")
    def delete_model(model_name):
        if not os.path.exists(dataset_manager.model_spec_path(model_name)):
            return f"Model '{model_name}' does not exist", 400
        
        with open(dataset_manager.model_spec_path(model_name), "r") as file:
            model_spec = json.load(file)
            
        os.remove(dataset_manager.model_spec_path(model_name))
        if os.path.exists(dataset_manager.model_preds_path(model_name)):
            os.remove(dataset_manager.model_preds_path(model_name))
        if os.path.exists(dataset_manager.model_metrics_path(model_name)):
            os.remove(dataset_manager.model_metrics_path(model_name))
        if os.path.exists(dataset_manager.model_weights_path(model_name)):
            os.remove(dataset_manager.model_weights_path(model_name))
            
        with evaluator_lock:
            if evaluator is not None:
                # Mark that the model's metrics have changed
                evaluator.rescore_model(model_name, model_spec["timestep_definition"])
        return "Success"

    @app.route("/models/<model_name>/metrics")
    def get_model_metrics(model_name):
        if not os.path.exists(dataset_manager.model_spec_path(model_name)):
            return f"Model '{model_name}' does not exist", 400
        
        if not os.path.exists(os.path.join(MODEL_DIR, f"metrics_{model_name}.json")):
            return f"No metrics available", 400
        with open(dataset_manager.model_spec_path(model_name), "r") as file:
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
        if "draft" in body:
            with open(dataset_manager.model_spec_path(model_name), "r") as file:
                meta = json.load(file)
                
            # Just save the model spec as a draft
            with open(dataset_manager.model_spec_path(model_name), "w") as file:
                if not len(body["draft"]):
                    # Delete the draft
                    if "draft" in meta: del meta["draft"]
                    json.dump(meta, file)
                else:
                    json.dump({**meta, "draft": body["draft"]}, file)
                    
            return "Draft saved"
        
        if "meta" not in body:
            return "Model 'meta' key required", 400
        meta = body["meta"]
        if not meta.get("timestep_definition", None):
            return "Timestep definition is required", 400
        if not meta.get("outcome", None):
            return "Outcome is required", 400
        
        with model_worker_comm_lock:
            if _is_model_training():
                with open(dataset_manager.model_spec_path(model_name), "w") as file:
                    json.dump({**meta, "training": True, "status": { 
                        "state": "waiting", 
                        "message": "Waiting for other training jobs to complete"
                    }}, file)
            else:
                with open(dataset_manager.model_spec_path(model_name), "w") as file:
                    json.dump({**meta, "training": True, "status": { 
                        "state": "loading", 
                        "message": "Starting"
                    }}, file)
            queue.put(("train_model", model_name, meta))
            
        with evaluator_lock:
            if evaluator is not None:
                # Mark that the model's metrics have changed
                evaluator.rescore_model(model_name, meta["timestep_definition"])
        return f"Started training model '{model_name}'", 200
        
    data_summary = None
    
    @app.route("/data/summary")
    def get_data_summary():
        global data_summary
        if data_summary is not None: return jsonify(data_summary)
        result = {}
        result["attributes"] = {attr_name: make_query_result_summary(sample_dataset, sample_dataset.attributes.get(attr_name))
                                for attr_name in sample_dataset.attributes.df.columns}
        result["events"] = {eventtype: make_query_result_summary(sample_dataset, sample_dataset.events.get(eventtype))
                            for eventtype in sample_dataset.events.get_types().unique()}
        result["intervals"] = {eventtype: make_query_result_summary(sample_dataset, sample_dataset.intervals.get(eventtype))
                            for eventtype in sample_dataset.intervals.get_types().unique()}
        data_summary = sf.utils.convert_to_native_types(result)
        return jsonify(data_summary)
        
    @app.route("/data/fields")
    def list_data_fields():
        result = [
            *sample_dataset.attributes.df.columns,
            *sample_dataset.events.get_types().unique(),
            *sample_dataset.intervals.get_types().unique()
        ]
        return jsonify({"fields": sf.utils.convert_to_native_types(result)})
        
    @app.route("/data/query")
    def query_dataset():
        args = request.args
        if "q" not in args: return "Query endpoint must have a 'q' query argument", 400
        try:
            if args.get("dl", "0") == "1":
                dataset, (train_ids, val_ids, test_ids) = dataset_manager.load_data(sample=SAMPLE_MODEL_TRAINING_DATA)
                result = dataset.query(args.get("q"))
                memory_file = BytesIO()
                with zipfile.ZipFile(memory_file, 'w') as zf:
                    data = zipfile.ZipInfo(f'query.txt')
                    data.date_time = time.localtime(time.time())[:6]
                    data.compress_type = zipfile.ZIP_DEFLATED
                    zf.writestr(data, args.get("q"))
                    for filename, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
                        data = zipfile.ZipInfo(f'{filename}.csv')
                        data.date_time = time.localtime(time.time())[:6]
                        data.compress_type = zipfile.ZIP_DEFLATED
                        zf.writestr(data, result.filter(result.get_ids().isin(ids)).to_csv())
                memory_file.seek(0)
                return send_file(memory_file, 
                                 as_attachment=True, 
                                 download_name=f"query_result_{str(datetime.datetime.now().replace(microsecond=0))}.zip")
            else:
                result = sample_dataset.query(args.get("q"), use_cache=False)
                summary = {
                    "n_values": len(result.get_values()),
                    "n_trajectories": len(set(result.get_ids().values.tolist())),
                    "result_type": QUERY_RESULT_TYPENAMES.get(type(result), "Other"),
                    "query": args.get("q")
                }
                return jsonify({**summary, "result": make_query_result_summary(sample_dataset, result)})
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return jsonify({"error": str(e)})
        
    @app.post("/data/download")
    def download_batch_queries():
        body = request.json
        if "queries" not in body: return "/data/download request body must include a queries dictionary", 400
        try:
            dataset, (train_ids, val_ids, test_ids) = dataset_manager.load_data(sample=SAMPLE_MODEL_TRAINING_DATA)
            memory_file = BytesIO()
            with zipfile.ZipFile(memory_file, 'w') as zf:
                data = zipfile.ZipInfo(f'queries.txt')
                data.date_time = time.localtime(time.time())[:6]
                data.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(data, "\n\n".join(f"{name}\n{query}" for name, query in body.get("queries").items()))
                for query_name, query in body.get("queries").items():
                    result = dataset.query(query)
                    for filename, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
                        data = zipfile.ZipInfo(f'{query_name}_{filename}.csv')
                        data.date_time = time.localtime(time.time())[:6]
                        data.compress_type = zipfile.ZIP_DEFLATED
                        zf.writestr(data, result.filter(result.get_ids().isin(ids)).to_csv())
            memory_file.seek(0)
            if "filename" in body:
                filename = body.get("filename") + ".zip"
            else:
                filename = f"query_result_{str(datetime.datetime.now().replace(microsecond=0))}.zip"
            return send_file(memory_file, 
                                as_attachment=True, 
                                download_name=filename)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return jsonify({"error": str(e)})
        
    @app.post("/data/validate_syntax")
    def validate_syntax():
        body = request.json
        if "query" not in body: return "validate_syntax request body must include a query", 400
        try:
            query = body.get("query")
            result = sample_dataset.parse(query, keep_all_tokens=True)
            
            unnamed_index = 0
            parsed_variables = {}
            for var_exp in result.find_data("variable_expr"):
                if len(list(var_exp.find_data("variable_expr"))) > 1: continue
                var_name_node = next(var_exp.find_data("named_variable"), None)
                if var_name_node: var_name = var_name_node.children[0].value
                else: 
                    var_name = "Unnamed " + (str(unnamed_index) if unnamed_index > 0 else "")
                    unnamed_index += 1
                min_pos = min(token.start_pos for token in var_exp.children[-1].scan_values(lambda x: isinstance(x, lark.Token)))
                max_pos = max(token.end_pos for token in var_exp.children[-1].scan_values(lambda x: isinstance(x, lark.Token)))
                print(var_name, query[min_pos:max_pos])
                parsed_variables[var_name] = {"query": query[min_pos:max_pos], "enabled": True}
            return jsonify({"success": True, "variables": parsed_variables})
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return jsonify({"success": False, "error": str(e)})
        
        
    @app.route("/models/status/<model_name>")
    def model_training_status(model_name):
        return jsonify(_get_model_training_status(model_name))
        
    @app.route("/models/stop_training/<model_name>", methods=["POST"])
    def stop_model_training(model_name):
        global model_worker, queue
        state = _get_model_training_status(model_name)["state"]
        if state == "loading":
            with model_worker_comm_lock:
                os.kill(model_worker.pid, signal.SIGINT)
                print("Restarting model worker")
                queue = mp.Queue()
                model_worker = mp.Process(target=_background_model_generation, args=(base_path, queue), kwargs={"single_thread": args.single_thread})
                model_worker.start()
        elif state in ("waiting", "error"):
            with open(dataset_manager.model_spec_path(model_name), "r") as file:
                meta = json.load(file)
            with open(dataset_manager.model_spec_path(model_name), "w") as file:
                json.dump({k: v for k, v in meta.items() if k not in ("training", "status")}, file)
        return "Success"
        
    @app.route("/slices/<model_name>/start", methods=["POST"])
    def start_slice_finding(model_name):
        if os.path.exists(os.path.join(MODEL_DIR, f"slices_{model_name}.json")):
            with open(os.path.join(MODEL_DIR, f"slices_{model_name}.json"), "r") as file:
                slice_progress = json.load(file)
                if slice_progress.get("searching", False) and slice_progress["status"]["state"] != "error":
                    return "Already finding slices", 400
        else:
            slice_progress = {}
    
        if not os.path.exists(dataset_manager.model_spec_path(model_name)):
            return "Model does not exist", 404
        
        with open(dataset_manager.model_spec_path(model_name), "r") as file:
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
        with model_worker_comm_lock:
            queue.put(("find_slices", model_name, controls))
            search_status = {"state": "loading", "message": "Starting", "model_name": model_name}
            finder = dataset_manager.make_slice_discovery_helper()
            finder.write_status(True, search_status=search_status)
        return jsonify(finder.get_status())
    
    @app.route("/slices/stop_finding", methods=["POST"])
    def stop_slice_finding():
        global model_worker, queue
        with model_worker_comm_lock:
            os.kill(model_worker.pid, signal.SIGINT)
            print("Restarting model worker")
            queue = mp.Queue()
            model_worker = mp.Process(target=_background_model_generation, args=(base_path, queue), kwargs={"single_thread": args.single_thread})
            model_worker.start()
        return "Success"
      
    @app.route("/slices/status")
    def get_slice_status():
        slice_finder = dataset_manager.make_slice_discovery_helper()
        status = slice_finder.get_status()
        return jsonify(status)

    @app.route("/slices/<model_names>", methods=["POST"])
    def get_slices(model_names):
        model_names = model_names.split(",")
        
        timestep_def = None
        for name in model_names:
            with open(dataset_manager.model_spec_path(name), "r") as file:
                spec = json.load(file)
            if spec.get("training", False):
                return "Cannot get slices for model that isn't finished training", 400
            if not os.path.exists(dataset_manager.model_metrics_path(name)):
                return "Cannot get slices for model with no metrics", 400
            if timestep_def is not None and spec["timestep_definition"] != timestep_def:
                return "Cannot get slices for models with multiple timestep definitions", 400
            timestep_def = spec["timestep_definition"]
            
        with evaluator_lock:
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
        ranked = rank_list.rank(weights, n_slices=body.get("num_slices", 20))
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
        
    def _score_slice(model_names, timestep_def, slice_spec_name, slice_requests, return_instance_info=False):
        rank_list, metrics, ids, df = evaluator.get_slice_ranking_info(timestep_def, {"slice_spec_name": slice_spec_name}, model_names)
        
        slices_to_score = {k: rank_list.encode_slice(v) for k, v in slice_requests.items()}
        scored_slices = {
            k: evaluator.describe_slice(rank_list,
                                        metrics, 
                                        ids,
                                        slice_obj.rescore(rank_list.score_slice(slice_obj)),
                                        model_names,
                                        return_instance_info=True)
            for k, slice_obj in slices_to_score.items()
        }
        masks = {k: v[1] for k, v in scored_slices.items()}
        scored_slices = sf.utils.convert_to_native_types({k: v[0] for k, v in scored_slices.items()})
        if return_instance_info:
            return scored_slices, masks
        return scored_slices
        
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
                    results[slice_key]["scoreValues"].update(ts_results[slice_key]["scoreValues"])
                    results[slice_key]["metrics"].update(ts_results[slice_key]["metrics"])
                # if "selectedModel" in body and body["selectedModel"] in model_names:
                #     slice_mask = instance_infos[slice_key][0].astype(bool)
                #     selected_true, selected_pred = instance_infos[slice_key][1][body["selectedModel"]]
                #     selected_mask = (~pd.isna(selected_true))[slice_mask]
                #     # What does the venn diagram color mean in this context? are the circles different models?
                #     for model_name in model_names:
                #         model_true, model_pred = instance_infos[slice_key][1][model_name]
                #         model_mask = (~pd.isna(model_true))[slice_mask]
                #         def make_overlap_metric(mask, values):
                #             return sf.utils.convert_to_native_types({
                #                 "count": mask.sum(), 
                #                 "mean": values[mask].mean()
                #             })
                        
                #         results[slice_key]["metrics"][model_name]["Overlap True"] = {
                #             "type": "overlap",
                #             "neither": make_overlap_metric((~selected_mask) & (~model_mask), model_true),
                #             "left": make_overlap_metric(selected_mask & (~model_mask), model_true),
                #             "right": make_overlap_metric(~selected_mask & model_mask, model_true),
                #             "both": make_overlap_metric(selected_mask & model_mask, model_true),
                #         }
                #         results[slice_key]["metrics"][model_name]["Overlap Pred"] = {
                #             "type": "overlap",
                #             "neither": make_overlap_metric((~selected_mask) & (~model_mask), model_pred),
                #             "left": make_overlap_metric(selected_mask & (~model_mask), model_pred),
                #             "right": make_overlap_metric(~selected_mask & model_mask, model_pred),
                #             "both": make_overlap_metric(selected_mask & model_mask, model_pred),
                #         }
                    
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
        
        discrete_df, ids, slice_filter = evaluator.get_slicing_data(body.get("sliceSpec", "default"),
                                                                 timestep_def,
                                                                 evaluation=True)
        slice_to_eval = discrete_df.encode_slice(body["slice"])
        
        offset = body.get("offset", None)
        
        valid_mask = np.all(np.vstack([evaluator.get_valid_model_mask(n) for n in model_names]), axis=0)
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
        slice_spec_dir = os.path.join(dataset_manager.slices_dir, "specifications")
        results = {}
        if os.path.exists(slice_spec_dir):
            for path in os.listdir(slice_spec_dir):
                if not path.endswith(".json"): continue
                spec_name = os.path.splitext(path)[0]
                with open(os.path.join(slice_spec_dir, path), "r") as file:
                    results[spec_name] = json.load(file)
        results["default"] = dataset_manager.get_default_slice_spec()
        return jsonify(results)
    
    @app.route("/slices/specs/<spec_name>", methods=["POST"])
    def edit_slice_spec(spec_name):
        slice_spec_dir = os.path.join(dataset_manager.slices_dir, "specifications")
        if not os.path.exists(slice_spec_dir):
            os.mkdir(slice_spec_dir)
        body = request.json
        if "variables" not in body or "slice_filter" not in body:
            return "Slice spec must contain 'variables' and 'slice_filter' keys", 400
        
        dataset = evaluator.get_raw_slicing_dataset()
        for var_name, var in body["variables"].items():
            try:
                dataset.parser.parse(var['query'])
            except Exception as e:
                return f"Error parsing variable {var_name}: {e}", 400
        
        spec_path = os.path.join(slice_spec_dir, f"{spec_name}.json")
        if os.path.exists(spec_path):
            print(f"Invalidating existing slices with spec {spec_name}")
            with model_worker_comm_lock:
                queue.put(("invalidate_slice_spec", spec_name))
            with evaluator_lock:
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
    
    if args.profile:
        app.wsgi_app = ProfilerMiddleware(app.wsgi_app, sort_by=["cumtime"], restrictions=[100])
    app.run(debug=not args.public, port=args.port, host='0.0.0.0' if args.public else '127.0.0.1')