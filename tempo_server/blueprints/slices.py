from flask import Blueprint, jsonify, request
from flask_login import login_required
from ..compute.run import get_worker, get_filesystem
from ..compute.utils import Commands
from ..compute.dataset import Dataset
from ..compute.slicefinder import SliceFinder
from ..compute.utils import make_series_summary
from divisi.utils import convert_to_native_types
import traceback
import numpy as np

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
@login_required
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
    
    try:
        if dataset_name not in slice_evaluators:
            slice_evaluators[dataset_name] = SliceFinder(dataset)
        slice_evaluator = slice_evaluators[dataset_name]
        results = slice_evaluator.lookup_slice_results(
            model_name,
            body['variable_spec_name'],
            body['score_function_spec']
        )
        
        timestep_def = dataset.get_model(model_name).get_spec()['timestep_definition']
        evaluation_results = slice_evaluator.evaluate_slices(
            results if results else [], 
            timestep_def, 
            body['variable_spec_name'],
            body.get('model_names', [model_name]),
            score_function_spec=body['score_function_spec'],
            include_meta=True
        )
        return jsonify(convert_to_native_types(evaluation_results))
    except Exception as e:
        traceback.print_exc()
        return f"Error occurred while retrieving subgroups: {str(e)}", 500
    
@slices_blueprint.post("/datasets/<dataset_name>/slices/<model_name>/find")
@login_required
def start_slice_finding(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: name of the dataset in which to search for slices
    * model_name: name of the model that sets the timestep definition for the slices
    
    Request body: JSON of the format {
        "variable_spec_name": name of a slice spec,
        "score_function_spec": list of score function objects (see below), 
        "options"?: {
            "rule_filter"?: a rule filter, specified in the format below.
            "num_samples"?: number of samples to draw (default 100),
            "num_slices"?: number of slices to return (default 20),
            "min_items_fraction"?: fraction of dataset to allow as minimum slice size (default 0.05),
        }
    }
    Each score function should be an ScoreExpression JSON of the format: {
        "type": "model_property",
        "model_name": string, 
        "property": "label" | "prediction" | "correctness" | "deviation" | "abs_deviation"
    } | { "type": "constant", "value": value } | { 
        "type": "relation", 
        "relation": "=" | "!=" | "<" | "<=" | ">" | ">=" | "in" | "not-in",
        "lhs": ScoreExpression,
        "rhs": ScoreExpression
    } | { 
        "type": "logical", 
        "relation": "and" | "or",
        "lhs": ScoreExpression,
        "rhs": ScoreExpression
    }
    
    The rule filter should be in the following format: {
        "type": "combination",
        "combination": "and" | "or",
        "logic": "exclude" | "include",
        "lhs": RuleFilter,
        "rhs": RuleFilter
    } | {
        "type": "constraint",
        "logic": "exclude" | "include",
        "features": [<array of user-readable feature names>],
        "values": [<array of user-readable values>]
    }
    
    Returns: A JSON task info object representing the status of the slice
    finding task. Use /datasets/<ds>/slices/<model> to fetch the results.
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
    
@slices_blueprint.post("/datasets/<dataset_name>/slices/<model_name>/validate_score_function")
@login_required
def validate_score_function_spec(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: name of the dataset in which to define the score function
    * model_name: name of the model used as the base for the score function
    
    Request body: JSON of the format {
        "score_function": ScoreExpression (as defined in documentation for 
            /datasets/<dataset_name>/slices/<model_name>/find)
    }
    
    Returns: JSON of the format {
        "result": QueryResult
    } if successful, otherwise { "error": string }
    """
    global slice_evaluators
    
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name), "test")
    if not dataset.fs.exists():
        return "Dataset does not exist", 404

    body = request.json
    if 'score_function' not in body:
        return "Request body must include 'score_function' key", 400
    
    try:
        if dataset_name not in slice_evaluators:
            slice_evaluators[dataset_name] = SliceFinder(dataset)
        slice_evaluator = slice_evaluators[dataset_name]
        
        eval_data = slice_evaluator.parse_score_expression(body['score_function'], 'test')
        uniques = np.unique(eval_data[~np.isnan(eval_data)]).astype(int)
        assert len(uniques) <= 2 and not (set(uniques) - set([0, 1])), "Score functions must result in a binary value"
        return jsonify(convert_to_native_types({"result": {"values": make_series_summary(eval_data)}}))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})
    
@slices_blueprint.post("/datasets/<dataset_name>/slices/<model_name>/score")
@login_required
def score_slice(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: name of the dataset in which to score the slice
    * model_name: name of the model used as the base model (for timestep definition)
    
    Request body: JSON of the format {
        "variable_spec_name": name of the variable spec,
        "score_function_spec": score function to get optional score criterion metric,
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
        score_function_spec=body.get("score_function_spec", None),
        encode_slices=True
    )
    return jsonify({ "slices": convert_to_native_types(evaluation_results) })

@slices_blueprint.route("/datasets/<dataset_name>/slices/<model_name>/compare", methods=["POST"])
@login_required
def get_slice_comparisons(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: name of the dataset in which to score the slice
    * model_name: name of the model used as the base model (for timestep definition)
    
    Request body: JSON of the format {
        "slice": <slice feature to compare>,
        "offset" (optional): number of timesteps forward (positive) or backward
            (negative) to compare against
        "variable_spec_name": name of the variable spec to use
    }
    
    Returns: if offset is provided, a JSON of the format {
        "top_changes": [
            { "variable": var name, "enrichments": [
                {
                    "source_value": any,
                    "destination_value": any 
                    "base_prob": number,
                    "slice_prob": number,
                    "ratio": number
                }, ..
            ] },
            ...
        ], 
        "source": {
            "top_variables": [
                { 
                    "variable": var name, "enrichments": [
                        { "value": any, "ratio": number },
                        ...
                    ] 
                }, ...
            ],
            "all_variables": {
                <var name>: {
                    "values": string[],
                    "base": number[] (counts in overall dataset),
                    "slice": number[] (counts in slice),
                }
            }
        },
        "destination": same as source
    }. If offset is not provided, returns a JSON of the format {
        "top_variables": [
            { 
                "variable": var name, "enrichments": [
                    { "value": any, "ratio": number },
                    ...
                ] 
            }, ...
        ],
        "all_variables": {
            <var name>: {
                "values": string[],
                "base": number[] (counts in overall dataset),
                "slice": number[] (counts in slice),
            }
        }
    }
    """
        
    global slice_evaluators
    
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name), "test")
    if not dataset.fs.exists():
        return "Dataset does not exist", 404

    body = request.json
    if 'variable_spec_name' not in body:
        return "Request body must include 'variable_spec_name' key", 400
    if "slice" not in body:
        return "'slice' body argument required", 400
    
    offset = body.get("offset", None)
    variable_spec_name = body.get("variable_spec_name")
    
    if dataset_name not in slice_evaluators:
        slice_evaluators[dataset_name] = SliceFinder(dataset)
    slice_evaluator = slice_evaluators[dataset_name]

    if offset is not None:
        try:
            if int(offset) != offset:
                return "Offset must be an integer", 400
        except ValueError:
            return "Offset must be an integer", 400
        
        # Return a comparison of the slice at the given number of steps
        # offset from the current time compared to the current time
        comparison = slice_evaluator.describe_slice_change_differences(offset, 
                                                                       body.get('slice'), 
                                                                       model_name, 
                                                                       variable_spec_name)
        return jsonify(convert_to_native_types(comparison))
    else:
        differences = slice_evaluator.describe_slice_differences(
            body.get('slice'),
            model_name,
            variable_spec_name
        )
        return jsonify(convert_to_native_types(differences))
    
@slices_blueprint.route("/datasets/<dataset_name>/slices/specs", methods=["GET"])
@login_required
def get_slice_specs(dataset_name):
    """
    Parameters:
    * dataset_name: name of the dataset in which to get the slice specs
    
    Returns: JSON of the format {
        spec name: <spec>,
        spec name: <spec>,
        ...
    }
    """
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name), "test")
    if not dataset.fs.exists():
        return "Dataset does not exist", 404

    specs = dataset.get_slicing_variable_specs()

    return jsonify({spec_name: spec.get_spec() for spec_name, spec in specs.items()})

@slices_blueprint.get("/datasets/<dataset_name>/slices/specs/<spec_name>")
@login_required
def get_slice_spec(dataset_name, spec_name):
    """
    Parameters:
    * dataset_name: name of the dataset in which to search for the slice spec
    * spec_name: name of the spec to retrieve
    
    Returns: JSON representing the slice spec, or an error message if the slice
    spec does not exist.
    """
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name), "test")
    if not dataset.fs.exists():
        return "Dataset does not exist", 404

    spec = dataset.get_slicing_variable_spec(spec_name)
    try:
        return jsonify(spec.get_spec())
    except:
        return "Slicing spec does not exist", 404
        

@slices_blueprint.post("/datasets/<dataset_name>/slices/specs/<spec_name>")
@login_required
def edit_slice_spec(dataset_name, spec_name):
    """
    Parameters:
    * dataset_name: name of the dataset in which to search for the slice spec
    * spec_name: name of the spec to edit (does not need to exist beforehand)
    
    Request body: JSON representing the slice spec.
    
    Returns: a plain-text success message if the slice spec was edited.
    """
    global slice_evaluators
    
    dataset = Dataset(get_filesystem().subdirectory("datasets", dataset_name), "test")
    if not dataset.fs.exists():
        return "Dataset does not exist", 404

    spec = dataset.get_slicing_variable_spec(spec_name)
    spec.write_spec(request.json)

    if dataset_name in slice_evaluators:
        slice_evaluator = slice_evaluators[dataset_name]
        slice_evaluator.invalidate_variable_spec(spec_name)

    return "Success"