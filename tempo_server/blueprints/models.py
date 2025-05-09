from flask import Blueprint, jsonify, request
from flask_login import login_required
from ..compute.run import get_worker, get_filesystem
from ..compute.dataset import Dataset
from ..compute.utils import Commands
from ..compute.model import Model
from ..compute.worker import TaskStatus
from .slices import slice_evaluators

models_blueprint = Blueprint('models', __name__)

def is_training_model(dataset_name, model_name):
    """
    Returns True if the model is being trained in a waiting or running job.
    """
    worker = get_worker()
    return any(
        w['info']['cmd'] == Commands.TRAIN_MODEL 
        and w['info']['dataset_name'] == dataset_name
        and w['info']['model_name'] == model_name
        for w in worker.current_jobs()
    )
    
def model_training_job_info(dataset_name, model_name):
    """
    Returns an info dictionary of the format {
        'id': job id,
        'info': task info,
        'status': status (waiting, running, etc.),
        'status_info': an optional status message
    } if the model is training, or None otherwise.
    """
    worker = get_worker()
    return next((
        status for status in worker.current_jobs()
        if status['info']['cmd'] == Commands.TRAIN_MODEL 
        and status['info']['dataset_name'] == dataset_name
        and status['info']['model_name'] == model_name
    ), None)
    
@models_blueprint.route('/datasets/<dataset_name>/models', methods=["GET"])
@login_required
def get_models(dataset_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to return models
    
    Returns: JSON of the format { "models": {
        "model_1": { "spec": model_spec, "metrics": metrics },
        "model_2": { "spec": model_spec, "metrics": metrics },
        ...
    }}
    """
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name))
    try:
        contents = dataset.get_models()
    except Exception as e:
        print("Error listing models:", str(e))
        return f"Dataset does not exist", 404
    else:
        results = {}
        for m, model in contents.items():
            results[m] = { "spec": model.get_spec() }
            try:
                results[m]["metrics"] = model.get_metrics()
            except:
                pass
        return jsonify({ "models": results})

@models_blueprint.route("/datasets/<dataset_name>/models/<model_name>")
@login_required
def get_model_spec(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to return models
    * model_name: The name of the model to return the spec for
    
    Returns: JSON of the format {
        "name": "model name",
        "spec": { model spec }
    }
    """
    fs = get_filesystem().subdirectory("datasets", dataset_name, "models", model_name)
    print(fs.base_path)
    if not fs.exists():
        return f"Model does not exist", 404

    try:
        return jsonify({
            "name": model_name,
            "spec": Model(fs).get_spec()
        })
    except Exception as e:
        print('error reading spec:', e)
        return "Could not read spec", 400


@models_blueprint.post("/datasets/<dataset_name>/models/new/<reference_name>")
@login_required
def make_new_model_spec(dataset_name, reference_name=""):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to return models
    * reference_name: The name of the spec to duplicate to create the new
        spec. If "default" is provided, uses the default configured model
        spec for this dataset.
        
    Returns: JSON of the format {
        "name": <initial model name>,
        "spec": { model spec }
    }
    """
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name))
    
    if reference_name == "default":
        base_spec = dataset.default_model_spec()
        base_name = "Untitled"
        fs = get_filesystem().subdirectory("datasets", dataset_name)
    else:
        fs = get_filesystem().subdirectory("datasets", dataset_name)
        if not fs.exists():
            return "Dataset does not exist", 404
        
        try:
            base_spec = Model(fs.subdirectory("models", reference_name)).get_spec()
        except:
            return "Reference model does not exist", 404
        base_name = reference_name
        
    increment_index = None
    final_name = base_name
    while fs.exists("models", final_name):
        if increment_index is None:
            increment_index = 2
        else:
            increment_index += 1
        final_name = f"{base_name} {increment_index}"

    Model(dataset.model_spec_dir(final_name)).write_spec(base_spec)
    return jsonify({"name": final_name, "spec": base_spec})


@models_blueprint.delete("/datasets/<dataset_name>/models/<model_name>")
@login_required
def delete_model(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to delete the model
    * model_name: The name of the model to delete
    
    Returns: plain-text "Success" if the model was deleted
    """
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name))
    if not dataset.fs.exists():
        return "Dataset does not exist", 404
    
    model_dir = dataset.model_spec_dir(model_name)
    if not model_dir.exists():
        return "Model does not exist", 404
    try:
        model_dir.delete()
    except:
        return "Model could not be deleted", 400
    
    cache_dir = dataset.model_cache_dir(model_name)
    if cache_dir.exists():
        cache_dir.delete()

    return "Success"
    
@models_blueprint.post("/datasets/<dataset_name>/models/<model_name>/rename")
@login_required
def rename_model(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to delete the model
    * model_name: The name of the model to rename
    
    Request body: JSON of the format { 'name': new name }
    
    Returns: plain-text "Success" if the model was renamed
    """
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name))
    if not dataset.fs.exists():
        return "Dataset does not exist", 404
    
    body = request.json
    if "name" not in body:
        return "rename requires a 'name' field in the request body", 400
    
    model_dir = dataset.model_spec_dir(model_name)
    if not model_dir.exists():
        return "Model does not exist", 404
    try:
        model_dir.rename(dataset.model_spec_dir(body["name"]))
    except:
        return "Model could not be renamed", 400
    
    cache_dir = dataset.model_cache_dir(model_name)
    if cache_dir.exists():
        cache_dir.rename(dataset.model_cache_dir(body["name"]))

    return "Success"
    

@models_blueprint.route("/datasets/<dataset_name>/models/<model_name>/metrics")
@login_required
def get_model_metrics(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to find the model
    * model_name: The name of the model to return metrics for
    
    Returns: JSON containing the metrics for the given model. Returns a 400 error
        code if the model is being trained.
    """
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name))
    if not dataset.fs.exists():
        return f"Dataset does not exist", 404

    if is_training_model(dataset_name, model_name):
        return "Model is being trained", 400
    
    try:
        return jsonify(dataset.get_model(model_name).get_metrics())
    except Exception as e:
        print('error reading spec:', e)
        return "Metrics not available", 400

@models_blueprint.route("/datasets/<dataset_name>/models", methods=["POST"])
@login_required
def generate_model(dataset_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to find the model
    
    Request body: JSON of the format {
        "name": model name,
        "draft" (optional): { draft model spec },
        "spec" (optional): { model spec }
    }. Either "draft" or "spec" is required.
    
    Returns: If saving draft, a plain text success message. If training model,
        JSON of the task status representing the training task.
    """
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name))
    if not dataset.fs.exists():
        return "Dataset does not exist", 404

    body = request.json
    if "name" not in body:
        return "Model 'name' key required", 400
    model_name = body["name"]
    if "draft" in body:
        model = dataset.get_model(model_name)
        try:
            model.write_draft_spec(body["draft"])
        except Exception as e:
            print("Error writing draft:", e)
            return "Error saving draft", 400

        return "Draft saved"
    
    if "spec" not in body:
        return "Model 'spec' key required", 400
    spec = body["spec"]
    if not spec.get("timestep_definition", None):
        return "Timestep definition is required", 400
    if not spec.get("outcome", None):
        return "Outcome is required", 400
    
    # First save a draft of the model
    try:
        model = dataset.get_model(model_name)
        model.write_draft_spec(spec)
    except Exception as e:
        print("Error writing draft:", e)
        return "Error saving draft", 400    
    
    worker = get_worker()
    worker.submit_task({
        'cmd': Commands.TRAIN_MODEL,
        'dataset_name': dataset_name,
        'model_name': model_name,
        'spec': spec
    })
        
    if dataset_name in slice_evaluators:
        slice_evaluators[dataset_name].invalidate_model(model_name)
        slice_evaluators[dataset_name].invalidate_variable_spec(dataset.default_slicing_variable_spec_name(model_name))
        
    # with evaluator_lock:
    #     if evaluator is not None:
    #         # Mark that the model's metrics have changed
    #         evaluator.rescore_model(model_name, meta["timestep_definition"])
    return jsonify(model_training_job_info(dataset_name, model_name))

@models_blueprint.post("/datasets/<dataset_name>/models/<model_name>/instances")
@login_required
def get_model_instances(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to find the model
    * model_name: The name of the model to return metrics for
    
    Request body: JSON containing a field 'ids', containing numerical IDs of trajectories
        in the data.
    Returns: JSON with a "inputs" key whose value is a list of records
        for each input id and timestep. The predictions contain the following fields:
            "index": a dictionary of id and time for the record
            "inputs": a dictionary of input feature names to values
            "ground_truth": the true label for the model
        This request queues a background task if the data is not cached, in which
        case the response is a JSON object representing the task status.
    """
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name))
    if not dataset.fs.exists():
        return "Dataset does not exist", 404

    body = request.json
    
    if "ids" not in body:
        return "'ids' body field required", 400
    
    model = dataset.get_model(model_name)
    if not model.fs.exists():
        return "Model does not exist", 404    
    worker = get_worker()

    task_info = {
        'cmd': Commands.GET_MODEL_INSTANCES,
        'dataset_name': dataset_name,
        'model_name': model_name,
        'ids': body["ids"]
    }  
    matching_job = next((j for j in worker.all_jobs() if j['info'] == task_info), None)  
    if matching_job:
        if matching_job['status'] == TaskStatus.COMPLETE:
            return jsonify(matching_job['status_info'])
        return jsonify(matching_job)
    
    task_id = worker.submit_task(task_info)

    return jsonify(worker.task_info(task_id))

@models_blueprint.post("/datasets/<dataset_name>/models/<model_name>/predict")
@login_required
def get_model_predictions(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to find the model
    * model_name: The name of the model to return metrics for
    
    Request body: JSON containing either a field 'ids' or a field 'inputs'.
        If including 'ids', should be a list of numerical IDs of trajectories
        in the data. If 'inputs' is provided, should be a list of dictionaries
        where each key is an input feature in the model. (Use /datasets/<dataset_name>/models/<model_name>
        to get the spec, which contains the variable names in the "variables"
        field.) Pass a key "n_feature_importances" to specify the number of
        SHAP values to return for each instance (or 0 to skip computation of
        SHAP values).
    Returns: JSON with an "outputs" key whose value is a list of predictions
        for each input id or record. The predictions contain the following fields:
            "index": a dictionary of id and time for the record if using ids, or the
                original input dictionary if using inputs.
            "prediction": a float if the model is binary classification or
                regression, or a list of floats if multiclass.
            "ground_truth": the true label for the model if the instance was
                created by an ID.
            "feature_importances": a list of dictionaries with two keys: 'feature'
                and 'value'. This contains only the top 10 features.
        Returns a 400 error code if the model is being trained.
    """
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name))
    if not dataset.fs.exists():
        return "Dataset does not exist", 404

    body = request.json
    
    if ("ids" in body) == ("inputs" in body):
        return "Exactly one of 'ids' and 'inputs' required", 400
    
    model = dataset.get_model(model_name)
    if not model.fs.exists():
        return "Model does not exist", 404
    result = model.lookup_prediction_results(ids=body.get("ids", None), inputs=body.get("inputs", None), n_feature_importances=body.get("n_feature_importances", 5))
    if result:
        return jsonify({"result": result})
        
    worker = get_worker()

    task_info = {
        'cmd': Commands.RUN_MODEL_INFERENCE,
        'dataset_name': dataset_name,
        'model_name': model_name,
        **({'ids': body["ids"]} if 'ids' in body else {'inputs': body["inputs"]}),
        'n_feature_importances': body.get("n_feature_importances", 5)
    }  
    matching_job = next((j for j in worker.current_jobs() if j['info'] == task_info), None)  
    if matching_job:
        return jsonify(matching_job)
    
    task_id = worker.submit_task(task_info)

    return jsonify(worker.task_info(task_id))