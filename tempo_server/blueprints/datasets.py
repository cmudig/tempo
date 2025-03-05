from flask import Blueprint, jsonify, request
from ..compute.run import get_worker, get_filesystem, clear_sample_dataset
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
            if spec := ds.get_spec():
                results[n] = { "spec": spec, "models": list(ds.get_models().keys()) }
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

@datasets_blueprint.post("/datasets/new")
@datasets_blueprint.post("/datasets/new/<reference_name>")
def make_new_dataset(reference_name=None):
    """
    Parameters:
    * reference_name: The name of the spec to duplicate to create the new
        dataset. If none is provided, creates an empty dataset.
        
    Returns: JSON of the format {
        "name": <initial model name>,
        "spec": { model spec }
    }
    """
    if reference_name is None:
        base_spec = {
            "data": {
                "sources": [],
            },
            "slices": {
                "sampler": {
                    "min_items_fraction": 0.02,
                    "samples_per_model": 100,
                    "max_features": 3,
                    "scoring_fraction": 0.2,
                    "num_candidates": 20,
                    "similarity_threshold": 0.5
                }
            }
        }
        base_name = "Untitled"
    else:
        if not get_filesystem().subdirectory("datasets", reference_name).exists():
            return "Dataset does not exist", 404
        base_name = reference_name
        base_spec = Dataset(get_filesystem().subdirectory("datasets", reference_name)).get_spec()
        
    increment_index = None
    final_name = base_name
    while get_filesystem().subdirectory("datasets").exists(final_name):
        if increment_index is None:
            increment_index = 2
        else:
            increment_index += 1
        final_name = f"{base_name} {increment_index}"

    Dataset(get_filesystem().subdirectory("datasets", final_name)).write_spec(base_spec)
    return jsonify({"name": final_name, "spec": base_spec})

@datasets_blueprint.delete("/datasets/<dataset_name>")
def delete_model(dataset_name):
    """
    Parameters:
    * dataset_name: Name of the dataset to delete
    
    Returns: plain-text "Success" if the model was deleted
    """
    dataset_dir = get_filesystem().subdirectory("datasets", dataset_name)
    if not dataset_dir.exists():
        return "Dataset does not exist", 404
    
    try:
        dataset_dir.delete()
    except:
        return "Dataset could not be deleted", 400
    
    clear_sample_dataset()
    get_worker().submit_task({
        "cmd": Commands.CLEAR_MEMORY_CACHE
    })
    
    return "Success"
    
@datasets_blueprint.post("/datasets/<dataset_name>/rename")
def rename_dataset(dataset_name):
    """
    Parameters:
    * dataset_name: Name of the dataset to rename
    
    Request body: JSON of the format { 'name': new name }
    
    Returns: plain-text "Success" if the model was renamed
    """
    dataset_dir = get_filesystem().subdirectory("datasets", dataset_name)
    if not dataset_dir.exists():
        return "Dataset does not exist", 404
    
    body = request.json
    if "name" not in body:
        return "rename requires a 'name' field in the request body", 400
    
    try:
        dataset_dir.rename(get_filesystem().subdirectory("datasets", body.get("name")))
    except:
        return "Dataset could not be renamed", 400
    
    clear_sample_dataset()
    get_worker().submit_task({
        "cmd": Commands.CLEAR_MEMORY_CACHE
    })

    return "Success"
    
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