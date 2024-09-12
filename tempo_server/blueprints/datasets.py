from flask import Blueprint, jsonify, request
from ..compute.run import get_worker, get_filesystem
from ..compute.utils import Commands
from ..compute.dataset import Dataset
from ..compute.slicefinder import SliceFinder

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

@datasets_blueprint.post("/datasets/<dataset_name>/clear_cache")
def clear_caches(dataset_name):
    """
    Parameters:
    * dataset_name: The name of the dataset whose cache to clear
    
    Request body: JSON with format {
        "target": cache name to delete
    }
    Possible cache names: 'all' (clear the entire cache), 'variables' (all
    saved variables), 'models' (all model training results), 'slices' (all 
    slice results).
    
    Returns: A string success message if the operation was successful.
    """
    fs = get_filesystem().subdirectory("datasets")
    if not fs.exists(dataset_name, "spec.json"):
        return "Dataset not found", 404
    
    body = request.json
    if "target" not in body:
        return f"clear_cache endpoint requires a 'target' request body parameter", 400
    
    cache_to_delete = body["target"]    
    dataset = Dataset(fs.subdirectory(dataset_name))
    if cache_to_delete == 'all':
        dataset.global_cache_dir.delete()
    elif cache_to_delete == 'variables':
        dataset.get_variable_cache_fs().delete()
        SliceFinder(dataset).variable_cache_fs.delete()
        for split in ('train', 'val', 'test'):
            split_data = Dataset(fs.subdirectory(dataset_name), split=split)
            split_data.get_variable_cache_fs().delete()
            SliceFinder(split_data).variable_cache_fs.delete()
    elif cache_to_delete == 'models':
        dataset.global_cache_dir.subdirectory("models").delete()
    elif cache_to_delete == 'slices':
        dataset.global_cache_dir.subdirectory("slices").delete()
    else:
        return f"Unrecognized clear cache target '{cache_to_delete}'", 400
    
    return "Cache successfully cleared."