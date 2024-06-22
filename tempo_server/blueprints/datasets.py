from flask import Blueprint, jsonify
from ..compute.run import get_worker

datasets_blueprint = Blueprint('datasets', __name__)

# Dataset management

@datasets_blueprint.get("/datasets/<dataset_name>/spec")
def get_dataset_spec(dataset_name):
    """
    Parameters:
    * dataset_name: The name of the dataset whose spec to get
    
    Returns: JSON containing the spec of the dataset
    """
    pass

@datasets_blueprint.post("/datasets/<dataset_name>/spec")
def update_dataset_spec(dataset_name):
    """
    Parameters:
    * dataset_name: The name of the dataset whose spec to update
    
    Request body: JSON with format { "spec": { dataset spec } }
    
    Returns: Plain-text success message if successfully updated
    """
    pass

