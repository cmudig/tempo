import os
import uuid
import tempfile
import logging
from flask import Blueprint, jsonify, request
from flask_login import login_required
from werkzeug.utils import secure_filename
from ..compute.filesystem import GCSFilesystem
from ..compute.run import get_worker, get_filesystem, clear_sample_dataset
from ..compute.utils import Commands
from ..compute.dataset import Dataset
from ..compute.slicefinder import SliceFinder

datasets_blueprint = Blueprint('datasets', __name__)

# Dataset management

@datasets_blueprint.get("/datasets")
@login_required
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
@login_required
def get_dataset_spec(dataset_name):
    """
    Parameters:
    * dataset_name: The name of the dataset whose spec to get
    
    Returns: JSON with a 'spec' key containing the spec of the dataset
    """
    fs = get_filesystem()
    if not fs.exists("datasets", dataset_name, "spec.json"):
        return "Dataset not found", 404
    
    return jsonify({"spec": fs.read_file("datasets", dataset_name, "spec.json", format='json')})

@datasets_blueprint.post("/datasets/<dataset_name>/spec")
@login_required
def build_dataset(dataset_name):
    """
    Parameters:
    * dataset_name: The name of the dataset whose spec to update
    
    Request body: JSON with format { "spec": { dataset spec } }
    
    Returns: JSON of the format { "spec": spec } containing the dataset's new spec
    """
    body = request.json
    if "spec" not in body:
        return f"Update spec requires a request body with a 'spec' field", 400
    
    fs = get_filesystem()
    if not fs.exists("datasets", dataset_name, "spec.json"):
        return "Dataset not found", 404
    
    worker = get_worker()
    worker.submit_task({
        "cmd": Commands.CLEAR_MEMORY_CACHE
    })
    task_id = worker.submit_task({
        "cmd": Commands.BUILD_DATASET,
        "dataset_name": dataset_name,
        "spec": body["spec"]
    })
    
    clear_sample_dataset()
    
    return jsonify({ "task_id": task_id })

@datasets_blueprint.delete("/datasets/<dataset_name>/draft")
@login_required
def delete_draft_spec(dataset_name):
    """
    Parameters:
    * dataset_name: The name of the dataset whose draft spec to delete
    
    Returns: JSON with format { "success": True } if successful, or plain-text
        error message if status code is not 200
    """
    fs = get_filesystem()
    if not fs.exists("datasets", dataset_name, "spec.json"):
        return "Dataset not found", 404
    
    if not fs.exists("dataset_drafts", dataset_name, "spec.json"):
        return "No draft exists", 400
        
    fs.subdirectory("dataset_drafts", dataset_name).delete()
        
    return jsonify({ "success": True })

@datasets_blueprint.get("/datasets/<dataset_name>/draft")
@login_required
def get_dataset_draft_spec(dataset_name):
    """
    Parameters:
    * dataset_name: The name of the dataset whose spec to get
    
    Returns: JSON containing the current draft spec of the dataset, or an empty
    dictionary if no draft has been created
    """
    worker = get_worker()
    matching_task = next((t for t in worker.current_jobs() 
                         if t["info"]["cmd"] == Commands.BUILD_DATASET and t["info"]["dataset_name"] == dataset_name), None)
    if matching_task:
        return jsonify({"spec": matching_task["spec"], "build_task": matching_task["id"]})
    
    fs = get_filesystem()
    if not fs.exists("datasets", dataset_name, "spec.json"):
        return "Dataset not found", 404
    
    if fs.exists("dataset_drafts", dataset_name, "spec.json"):
        return jsonify({'spec': fs.read_file("dataset_drafts", dataset_name, "spec.json", format='json')})
    return jsonify({})

@datasets_blueprint.post("/datasets/<dataset_name>/draft")
@login_required
def update_dataset_draft_spec(dataset_name):
    """
    Parameters:
    * dataset_name: The name of the dataset whose draft spec to update
    
    Request body: JSON with format { "spec": { dataset spec } }
    
    Returns: JSON of the format { "spec": spec } containing the dataset's new draft spec
    """
    body = request.json
    if "spec" not in body:
        return f"Update spec requires a request body with a 'spec' field", 400
    
    fs = get_filesystem()
    if not fs.exists("datasets", dataset_name, "spec.json"):
        return "Dataset not found", 404
    
    dataset = Dataset(fs.subdirectory("dataset_drafts", dataset_name))
    if not dataset.fs.list_files():
        # first draft - copy existing files over
        fs.subdirectory("datasets", dataset_name).copy_directory_contents(dataset.fs)
    dataset.write_spec(body["spec"])
    
    for file_source in dataset.fs.list_files("data"):
        if not any(source['path'].split("/")[-1] == file_source 
                   for source in dataset.get_spec().get("data", {}).get("sources", [])):
            dataset.fs.delete("data", file_source)
            logging.info(f"Deleted extra file '{file_source}' from draft")
    
    return jsonify({ "spec": dataset.get_spec() })

@datasets_blueprint.post("/datasets/<dataset_name>/draft/add_source")
@login_required
def add_data_source(dataset_name):
    """
    Parameters:
    * dataset_name: The name of the dataset whose spec to update
    
    Request body: should contain a file with the new data source
    
    Returns: JSON of the format { "spec": spec } representing the new dataset spec
    """
    fs = get_filesystem()
    if not fs.exists("datasets", dataset_name, "spec.json"):
        return "Dataset not found", 404
    dataset = Dataset(fs.subdirectory("dataset_drafts", dataset_name))
    if not dataset.get_spec():
        dataset.write_spec(fs.read_file("datasets", dataset_name, "spec.json"))
    
    if 'newfile' not in request.files:
        return "No file provided", 400
    
    file = request.files['newfile']
    if not file or file.filename == '':
        return "The file was not found or has no name", 400
    if not os.path.splitext(file.filename)[-1] in ('.csv', '.arrow'):
        return "The file has an invalid extension", 400
    
    filename = secure_filename(file.filename)
    if not filename:
        filename = uuid.uuid4().hex + os.path.splitext(file.filename)[-1]
    else:
        filename = os.path.basename(filename)
        
    spec = dataset.get_spec()
    sources = spec.get("data", {}).get("sources", [])
    final_name = filename
    increment = 1
    while any(os.path.basename(s["path"]) == final_name for s in sources):
        increment += 1
        final_name = os.path.splitext(filename)[0] + f" {increment}" + os.path.splitext(filename)[1]
        
    with dataset.fs.open("data", final_name, mode='wb') as dest_file:
        file.save(dest_file)
        
    sources.append({
        "type": "",
        "path": f"data/{final_name}"
    })
    spec.setdefault('data', {})["sources"] = sources
    print("new spec", spec)
    dataset.write_spec(spec)
    
    return jsonify({ "spec": dataset.get_spec() })

@datasets_blueprint.post("/datasets/new")
@datasets_blueprint.post("/datasets/new/<reference_name>")
@login_required
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
            "description": "",
            "data": {
                "sources": [],
            },
            "slices": {
                "sampler": {
                    "min_items_fraction": 0.02,
                    "n_samples": 100,
                    "max_features": 3,
                    "scoring_fraction": 1.0,
                    "num_candidates": 20,
                    "similarity_threshold": 0.5,
                    "n_slices": 20
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
@login_required
def delete_dataset(dataset_name):
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
@login_required
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
@login_required
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