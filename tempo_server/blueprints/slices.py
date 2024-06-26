from flask import Blueprint, jsonify, request
from ..compute.run import get_worker, get_filesystem
from ..compute.utils import Commands

slices_blueprint = Blueprint('datasets', __name__)

# Slice discovery and evaluation

@slices_blueprint.route("/datasets/<dataset_name>/slices", methods=["POST"])
def get_slices(dataset_name):
    """
    Parameters:
    * dataset_name: name of the dataset in which to search for slices
    
    Request body: JSON of the format {
        "spec": name of a slice spec,
        "score_functions": list of score function objects (see below),
        "num_samples"?: number of samples to draw (default 100),
        "num_slices"?: number of slices to return (default 20),
        "min_items_fraction"?: fraction of dataset to allow as minimum slice size (default 0.05),
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
    
    Returns: if the slices are completed, a JSON of the format {
        "results": {
            "slices": array of slices,
            "base_slice": description of base slice,
            "value_names": dictionary of value names in discrete dataframe
        }
    }. Otherwise, a JSON task info object representing the status of the slice
    finding task.
    """
    pass
    
@slices_blueprint.route("/datasets/<dataset_name>/slices/<model_names>/score", methods=["POST"])
def score_slice(model_names):
    """
    Parameters:
    * dataset_name: name of the dataset in which to score the slice
    
    
    """
    pass

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
        