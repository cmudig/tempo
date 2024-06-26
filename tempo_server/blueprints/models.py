from flask import Blueprint, jsonify, request
from ..compute.run import get_worker, get_filesystem
from ..compute.utils import Commands
from ..compute.model import Model

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
    fs = get_filesystem().subdirectory("datasets", dataset_name)
    try:
        fs = fs.subdirectory("models")
    except:
        return f"Dataset is incorrectly formatted", 400
    try:
        contents = fs.list_files()
    except:
        return f"Dataset does not exist", 404
    else:
        results = {}
        for m in contents:
            model = Model(fs.subdirectory(m))
            results[m] = {
                "spec": model.get_spec()
            }
            try:
                results[m]["metrics"] = model.get_metrics()
            except:
                pass
        return jsonify({ "models": results})

@models_blueprint.route("/datasets/<dataset_name>/models/<model_name>")
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
    if reference_name == "default":
        base_spec = Model.blank_spec()
        base_name = "Untitled"
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

    fs.write_file(base_spec, "models", final_name, "spec.json")
    return jsonify({"name": final_name, "spec": base_spec})


@models_blueprint.delete("/datasets/<dataset_name>/models/<model_name>")
def delete_model(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to delete the model
    * model_name: The name of the model to delete
    
    Returns: plain-text "Success" if the model was deleted
    """
    fs = get_filesystem().subdirectory("datasets", dataset_name)
    if not fs.exists():
        return "Dataset does not exist", 404
    
    model_dir = fs.subdirectory("models", model_name)
    if not model_dir.exists():
        return "Model does not exist", 404
    try:
        model_dir.delete()
    except:
        return "Model could not be deleted", 400

    return "Success"
    
@models_blueprint.post("/datasets/<dataset_name>/models/<model_name>/rename")
def rename_model(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to delete the model
    * model_name: The name of the model to rename
    
    Request body: JSON of the format { 'name': new name }
    
    Returns: plain-text "Success" if the model was renamed
    """
    fs = get_filesystem().subdirectory("datasets", dataset_name)
    if not fs.exists():
        return "Dataset does not exist", 404
    
    body = request.json
    if "name" not in body:
        return "rename requires a 'name' field in the request body", 400
    
    model_dir = fs.subdirectory("models", model_name)
    if not model_dir.exists():
        return "Model does not exist", 404
    try:
        model_dir.rename(fs.subdirectory("models", body["name"]))
    except:
        return "Model could not be renamed", 400

    return "Success"
    

@models_blueprint.route("/datasets/<dataset_name>/models/<model_name>/metrics")
def get_model_metrics(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to find the model
    * model_name: The name of the model to return metrics for
    
    Returns: JSON containing the metrics for the given model. Returns a 400 error
        code if the model is being trained.
    """
    fs = get_filesystem().subdirectory("datasets", dataset_name, "models", model_name)
    if not fs.exists():
        return f"Model does not exist", 404

    if is_training_model(dataset_name, model_name):
        return "Model is being trained", 400
    
    try:
        return jsonify(Model(fs).get_metrics())
    except Exception as e:
        print('error reading spec:', e)
        return "Metrics not available", 400

@models_blueprint.route("/datasets/<dataset_name>/models", methods=["POST"])
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
    fs = get_filesystem().subdirectory("datasets", dataset_name)
    if not fs.exists():
        return "Dataset does not exist", 404

    body = request.json
    if "name" not in body:
        return "Model 'name' key required", 400
    model_name = body["name"]
    if "draft" in body:
        model = Model(fs.subdirectory("models", model_name))
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
        model = Model(fs.subdirectory("models", model_name))
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
        
    # with evaluator_lock:
    #     if evaluator is not None:
    #         # Mark that the model's metrics have changed
    #         evaluator.rescore_model(model_name, meta["timestep_definition"])
    return jsonify(model_training_job_info(dataset_name, model_name))
