from flask import Blueprint, jsonify

data_blueprint = Blueprint('data', __name__)

# Dataset info

@data_blueprint.route("/datasets/<dataset_name>/data/summary")
def get_data_summary(dataset_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to describe the data
    
    Returns: JSON of the format {
        "attributes": {
            "attr_1": QueryResult,
            "attr_2": QueryResult,
            ...
        },
        "events": {
            "event_1": QueryResult,
            "event_2": QueryResult,
            ...
        },
        "intervals": {
            "interval_1": QueryResult,
            "interval_2": QueryResult
        }
    }
    """
    return f"You asked for data summary for {dataset_name}"
    
@data_blueprint.route("/datasets/<dataset_name>/data/fields")
def list_data_fields(dataset_name):
    """
    Parameters:
    * dataset_name: Name of the dataset to describe
    
    Returns: JSON of the format [ "field_1", "field_2", ... ] where each
        field is the name of an attribute, event, or interval
    """
    pass
    
@data_blueprint.route("/datasets/<dataset_name>/data/query")
def query_dataset():
    """
    Parameters:
    * dataset_name: Name of the dataset to query
    * q (query parameter): Tempo query language string to query
    * dl (query parameter, optional): 1 to return a downloadable ZIP file
        containing the query results on each data split, 0 otherwise 
    
    Returns: If dl is 1, a downloadable ZIP file. Otherwise, a JSON of the 
        format {
            "n_values": number,
            "n_trajectories": number,
            "result_type": string,
            "query": string,
            "result": QueryResult
        } if query was run successfully, otherwise { "error": string }.
    """
    pass
    
@data_blueprint.post("/datasets/<dataset_name>/data/download")
def download_batch_queries():
    """
    Parameters:
    * dataset_name: Name of the dataset to query
    
    Request body: JSON of the format {
        "queries": {
            "query name 1": "query string",
            "query name 2": "query string",
            ...
        }
    }
    
    Returns: if query was run successfully, a downloadable ZIP file containing
        query results split into "query name 1_train.csv", "query name 2_val.csv",
        "query name 3_test.csv", "query name 2_train.csv", etc. If an error
        occurred, a JSON of the format { "error": string }.
    """
    pass
    
@data_blueprint.post("/datasets/<dataset_name>/data/validate_syntax")
def validate_syntax():
    """
    Parameters:
    * dataset_name: Name of the dataset to query
    
    Request body: JSON of the format { "query": string }
    
    Returns: JSON of the format { 
        "success": true, 
        "variables": {
            "var name": {
                "query": string,
                "enabled": boolean
            },
            ...
        }
    } if successful; otherwise, { "success": false, "error": string }
    """
    pass
    
