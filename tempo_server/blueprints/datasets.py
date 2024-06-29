from flask import Blueprint, jsonify, request
from ..compute.run import get_worker, get_filesystem
from ..compute.utils import Commands
from ..compute.dataset import Dataset

datasets_blueprint = Blueprint('datasets', __name__)

# Dataset management

@datasets_blueprint.get("/datasets")
def list_datasets():
    """
    Returns: JSON array of the format {
        dataset name: { "spec": dataset spec, "models": list of model names },
        ...
    }
    """
    fs = get_filesystem().subdirectory("datasets")
    results = {}
    for n in fs.list_files():
        try:
            ds = Dataset(fs.subdirectory(n))
            results[n] = { "spec": ds.get_spec(), "models": list(ds.get_models().keys()) }
        except:
            continue
    return jsonify(results)
    
@datasets_blueprint.get("/datasets/<dataset_name>/spec")
def get_dataset_spec(dataset_name):
    """
    Parameters:
    * dataset_name: The name of the dataset whose spec to get
    
    Returns: JSON containing the spec of the dataset
    """
    fs = get_filesystem()
    if not fs.exists("datasets", dataset_name, "spec.json"):
        return "Dataset not found", 404
    
    return fs.read_file("datasets", dataset_name, "spec.json", format='json')

@datasets_blueprint.post("/datasets/<dataset_name>/spec")
def update_dataset_spec(dataset_name):
    """
    Parameters:
    * dataset_name: The name of the dataset whose spec to update
    
    Request body: JSON with format { "spec": { dataset spec } }
    
    Returns: JSON of the format { "task_id": task id } representing the task
        for updating the dataset
    """
    body = request.json
    if "spec" not in body:
        return f"Update spec requires a request body with a 'spec' field", 400
    worker = get_worker()
    task_id = worker.submit_task({
        "cmd": Commands.BUILD_DATASET,
        "name": dataset_name,
        "spec": body["spec"]
    })

    return jsonify({ "task_id": task_id })