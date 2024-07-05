from flask import Blueprint, jsonify, request
from ..compute.run import get_worker, get_filesystem
from ..compute.utils import Commands
from ..compute.dataset import Dataset
from ..compute.slicefinder import SliceFinder
from divisi.utils import convert_to_native_types

slices_blueprint = Blueprint('slices', __name__)

# Slice discovery and evaluation

slice_evaluators = {}

def slice_finding_job_info(dataset_name, model_name, variable_spec_name, score_function_spec, options):
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
        if status['info']['cmd'] == Commands.FIND_SLICES 
        and status['info']['dataset_name'] == dataset_name
        and status['info']['model_name'] == model_name
        and status['info']['variable_spec_name'] == variable_spec_name
        and status['info']['score_function_spec'] == score_function_spec
        and status['info']['options'] == options
    ), None)

@slices_blueprint.post("/datasets/<dataset_name>/slices/<model_name>")    
def get_slice_finding_results(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: name of the dataset in which to search for slices
    * model_name: name of the model that sets the timestep definition for the slices
    
    Request body: same format as /datasets/<dataset_name>/models/<model_name>/find_slices,
        plus a model_names field listing the names of the models to use to calculate
        metrics. If this is not provided the results will be given for the base
        model name only.
    
    Returns: if the slices have already been found, a JSON object of the format {
        "slices": array of slices,
        "base_slice": information about the base slice,
        "value_names": value name transformation for the discrete dataframe
    }. If the slice finding is in progress, a JSON task info object representing
    the running task. Otherwise, an empty object.
    """
    global slice_evaluators
    
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name), "test")
    if not dataset.fs.exists():
        return "Dataset does not exist", 404

    body = request.json
    if 'variable_spec_name' not in body:
        return "Request body must include 'variable_spec_name' key", 400
    if 'score_function_spec' not in body:
        return "Request body must include 'score_function_spec' key", 400
    options = body.get('options', {})
    
    if (job_info := slice_finding_job_info(dataset_name, 
                                           model_name, 
                                           body['variable_spec_name'], 
                                           body['score_function_spec'], 
                                           options)) is not None:
        return jsonify(job_info)
    
    if dataset_name not in slice_evaluators:
        slice_evaluators[dataset_name] = SliceFinder(dataset)
    slice_evaluator = slice_evaluators[dataset_name]
    results = slice_evaluator.lookup_slice_results(
        model_name,
        body['variable_spec_name'],
        body['score_function_spec']
    )

    if not results: return jsonify({})
    
    timestep_def = dataset.get_model(model_name).get_spec()['timestep_definition']
    evaluation_results = slice_evaluator.evaluate_slices(
        results, 
        timestep_def, 
        body['variable_spec_name'],
        body.get('model_names', [model_name]),
        include_meta=True
    )
    return jsonify(convert_to_native_types(evaluation_results))
    
    
@slices_blueprint.post("/datasets/<dataset_name>/slices/<model_name>/find")
def start_slice_finding(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: name of the dataset in which to search for slices
    * model_name: name of the model that sets the timestep definition for the slices
    
    Request body: JSON of the format {
        "variable_spec_name": name of a slice spec,
        "score_function_spec": list of score function objects (see below),
        "options"?: {
            "num_samples"?: number of samples to draw (default 100),
            "num_slices"?: number of slices to return (default 20),
            "min_items_fraction"?: fraction of dataset to allow as minimum slice size (default 0.05),
        }
    }
    Each score function should be an ScoreExpression JSON of the format: {
        "type": "model_property",
        "model_name": string, 
        "property": "label" | "prediction" | "correctness" | "deviation"
    } | { "type": "constant", "value": value } | { 
        "type": "relation", 
        "relation": "=" | "!=" | "<" | "<=" | ">" | ">=" | "in" | "not-in",
        "lhs": ScoreExpression,
        "rhs": ScoreExpression
    }
    
    Returns: A JSON task info object representing the status of the slice
    finding task. Use /datasets/<ds>/models/<model>/slices to fetch the results.
    """
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name))
    if not dataset.fs.exists():
        return "Dataset does not exist", 404

    body = request.json
    if 'variable_spec_name' not in body:
        return "Request body must include 'variable_spec_name' key", 400
    if 'score_function_spec' not in body:
        return "Request body must include 'score_function_spec' key", 400
    options = body.get('options', {})
    
    worker = get_worker()
    task_id = worker.submit_task({
        "cmd": Commands.FIND_SLICES,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "variable_spec_name": body['variable_spec_name'],
        "score_function_spec": body['score_function_spec'],
        "options": options
    })    
    return jsonify(worker.task_info(task_id))
    
@slices_blueprint.post("/datasets/<dataset_name>/slices/<model_name>/score")
def score_slice(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: name of the dataset in which to score the slice
    * model_name: name of the model used as the base model (for timestep definition)
    
    Request body: JSON of the format {
        "variable_spec_name": name of the variable spe
        "slices": dictionary of slice IDs to slice features,
        "model_names": array of model names to get metrics for
    } 
    
    Returns: JSON of the format {
        "slices": dictionary of slice IDs to slice descriptions
    }
    
    """
    global slice_evaluators
    
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name), "test")
    if not dataset.fs.exists():
        return "Dataset does not exist", 404

    body = request.json
    if 'variable_spec_name' not in body:
        return "Request body must include 'variable_spec_name' key", 400
    if 'slices' not in body:
        return "Request body must include 'slices' key", 400
    
    if dataset_name not in slice_evaluators:
        slice_evaluators[dataset_name] = SliceFinder(dataset)
    slice_evaluator = slice_evaluators[dataset_name]

    timestep_def = dataset.get_model(model_name).get_spec()['timestep_definition']
    evaluation_results = slice_evaluator.evaluate_slices(
        body['slices'], timestep_def, body['variable_spec_name'], body.get('model_names', [model_name]),
        encode_slices=True
    )
    return jsonify({ "slices": convert_to_native_types(evaluation_results) })


@slices_blueprint.route("/datasets/<dataset_name>/slices/score", methods=["POST"])
def score_slice_all_models():
    pass

@slices_blueprint.route("/datasets/<dataset_name>/slices/<model_names>/compare", methods=["POST"])
def get_slice_comparisons(model_names):
    pass
    
    
@slices_blueprint.route("/datasets/<dataset_name>/slices/specs", methods=["GET"])
def get_slice_specs():
    pass

@slices_blueprint.route("/datasets/<dataset_name>/slices/specs/<spec_name>", methods=["POST"])
def edit_slice_spec(spec_name):
    pass
        