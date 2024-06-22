from flask import Blueprint, jsonify

models_blueprint = Blueprint('models', __name__)

@models_blueprint.route('/datasets/<dataset_name>/models', methods=["GET"])
def get_models(dataset_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to return models
    
    Returns: JSON of the format { "models": {
        "model_1": { model_spec },
        "model_2": { model_spec },
        ...
    }}
    """
    pass

@models_blueprint.route("/datasets/<dataset_name>/models/<model_name>/spec")
def get_model_definition(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to return models
    * model_name: The name of the model to return the spec for
    
    Returns: JSON representing the model spec
    """
    return f"get model definition for {dataset_name}, {model_name}"

@models_blueprint.post("/datasets/<dataset_name>/models/new/<reference_name>")
def make_new_model_spec(dataset_name, reference_name):
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
    pass

@models_blueprint.delete("/datasets/<dataset_name>/models/<model_name>")
def delete_model(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to delete the model
    * model_name: The name of the model to delete
    
    Returns: plain-text "Success" if the model was deleted
    """
    pass

@models_blueprint.route("/datasets/<dataset_name>/models/<model_name>/metrics")
def get_model_metrics(dataset_name, model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to find the model
    * model_name: The name of the model to return metrics for
    
    Returns: JSON containing the metrics for the given model
    """
    pass

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
    
    Returns: plain-text success message if model started training successfully.
    """
    pass
        
@models_blueprint.route("/datasets/<dataset_name>/models/status/<model_name>")
def model_training_status(model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to search for the model
    * model_name: Name of the model to check status of
    
    Returns: JSON of the format {
        "state": string,
        "message": string
    }
    """
    pass
    
@models_blueprint.route("/datasets/<dataset_name>/models/stop_training/<model_name>", methods=["POST"])
def stop_model_training(model_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to search for the model
    * model_name: Name of the model to stop training
    
    Returns: plain-text success message if stopped training
    """
    pass
            
