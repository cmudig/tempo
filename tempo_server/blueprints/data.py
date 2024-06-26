from flask import Blueprint, jsonify, request
from ..compute.run import get_filesystem, get_worker, get_sample_dataset
from ..compute.utils import Commands, QUERY_RESULT_TYPENAMES, make_query_result_summary
import base64
import datetime
import lark
from divisi.utils import convert_to_native_types

data_blueprint = Blueprint('data', __name__)

# Dataset info

def _get_data_summary_task(dataset_name):
    worker = get_worker()
    return next((
        status for status in worker.current_jobs()
        if status['info']['cmd'] == Commands.SUMMARIZE_DATASET 
        and status['info']['dataset_name'] == dataset_name
    ), None)

    
@data_blueprint.route("/datasets/<dataset_name>/data/summary")
def get_data_summary(dataset_name):
    """
    Parameters:
    * dataset_name: Name of the dataset in which to describe the data
    
    Returns: if precomputed, a JSON of the format {
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
    }. If not yet computed, a JSON of the task status representing the summarization task.
    """
    if (current_task := _get_data_summary_task(dataset_name)) is not None:
        return jsonify(current_task)
        
    fs = get_filesystem()
    if not fs.exists("datasets", dataset_name, "spec.json"):
        return "Dataset not found", 404

    dataset = get_sample_dataset(dataset_name)
    summary = dataset.get_summary()
    if summary is None:
        worker = get_worker()
        task_id = worker.submit_task({
            "cmd": Commands.SUMMARIZE_DATASET,
            "dataset_name": dataset_name
        })
        return jsonify(worker.task_info(task_id))

    return jsonify(summary)
    
@data_blueprint.route("/datasets/<dataset_name>/data/fields")
def list_data_fields(dataset_name):
    """
    Parameters:
    * dataset_name: Name of the dataset to describe
    
    Returns: JSON of the format [ "field_1", "field_2", ... ] where each
        field is the name of an attribute, event, or interval
    """
    sample_dataset = get_sample_dataset(dataset_name).dataset
    result = [
        *sample_dataset.attributes.df.columns,
        *sample_dataset.events.get_types().unique(),
        *sample_dataset.intervals.get_types().unique()
    ]
    return jsonify({"fields": convert_to_native_types(result)})
    
@data_blueprint.route("/datasets/<dataset_name>/data/query")
def query_dataset(dataset_name):
    """
    Parameters:
    * dataset_name: Name of the dataset to query
    
    Query parameters:
    * q: Tempo query language string to query
    * dl (optional): 1 to return a downloadable ZIP file
        containing the query results on each data split, 0 otherwise 
    
    Returns: If dl is 1:
        Returns a JSON of the format { "blob": base 64 encoded result } if the
        query job is complete. Otherwise, returns info about a running task to
        generate this downloadable file. 
    If dl is 0 (default), returns a JSON of the format {
            "n_values": number,
            "n_trajectories": number,
            "result_type": string,
            "query": string,
            "result": QueryResult
        } if query was run successfully, otherwise { "error": string }.
    """
    args = request.args
    if "q" not in args: return "Query endpoint must have a 'q' query argument", 400
    try:
        if args.get("dl", "0") == "1":
            dataset = get_sample_dataset(dataset_name)
            if (result_path := dataset.get_downloadable_query(args["q"])) is not None:
                contents = dataset.read_downloadable_query_result(result_path)
                return { 
                    "blob": base64.b64encode(contents).decode('ascii'), 
                    "filename": f"query_result_{str(datetime.datetime.now().replace(microsecond=0))}.zip" 
                }
            
            worker = get_worker()
            running_task = next((task for task in worker.current_jobs()
                                 if task['cmd'] == Commands.GENERATE_QUERY_DOWNLOAD
                                 and task['query'] == args["q"]), None)
            if running_task is not None:
                return jsonify(running_task)
            task_id = worker.submit_task({
                'cmd': Commands.GENERATE_QUERY_DOWNLOAD,
                'query': args['q'],
                'dataset_name': dataset_name
            })
            return jsonify(worker.task_info(task_id))
        else:
            sample_dataset = get_sample_dataset(dataset_name)
            result = sample_dataset.query(args.get("q"), use_cache=False)
            summary = {
                "n_values": len(result.get_values()),
                "n_trajectories": len(set(result.get_ids().values.tolist())),
                "result_type": QUERY_RESULT_TYPENAMES.get(type(result), "Other"),
                "query": args.get("q")
            }
            return jsonify({**summary, "result": make_query_result_summary(sample_dataset, result)})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)})

    
@data_blueprint.post("/datasets/<dataset_name>/data/download")
def download_batch_queries(dataset_name):
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
    body = request.json
    if "queries" not in body: return "/data/download request body must include a queries dictionary", 400
    dataset = get_sample_dataset(dataset_name)
    if (result_path := dataset.get_downloadable_query(body["queries"])) is not None:
        contents = dataset.read_downloadable_query_result(result_path)
        return { 
            "blob": base64.b64encode(contents).decode('ascii'), 
            "filename": f"query_result_{str(datetime.datetime.now().replace(microsecond=0))}.zip" 
        }
    
    worker = get_worker()
    running_task = next((task for task in worker.current_jobs()
                            if task['cmd'] == Commands.GENERATE_QUERY_DOWNLOAD
                            and task['query'] == body["queries"]), None)
    if running_task is not None:
        return jsonify(running_task)
    task_id = worker.submit_task({
        'cmd': Commands.GENERATE_QUERY_DOWNLOAD,
        'query': body['queries'],
        'dataset_name': dataset_name
    })
    return jsonify(worker.task_info(task_id))


    
@data_blueprint.post("/datasets/<dataset_name>/data/validate_syntax")
def validate_syntax(dataset_name):
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
    body = request.json
    if "query" not in body: return "validate_syntax request body must include a query", 400
    try:
        query = body.get("query")
        sample_dataset = get_sample_dataset(dataset_name).dataset
        result = sample_dataset.parse(query, keep_all_tokens=True)
        
        unnamed_index = 0
        parsed_variables = {}
        ignore_exps = set() # for inner variable expressions
        top_variable_list = next((n for n in result.iter_subtrees_topdown() if isinstance(n, lark.Tree) and n.data == "variable_list"), None)
        if top_variable_list is None:
            raise ValueError("No variable list defined")
        for var_exp in top_variable_list.iter_subtrees_topdown():
            if var_exp in ignore_exps or not isinstance(var_exp, lark.Tree) or not var_exp.data == "variable_expr": continue
            for child in var_exp.find_data("variable_expr"):
                ignore_exps.add(child)
            var_name_node = next(var_exp.find_data("named_variable"), None)
            if var_name_node: var_name = var_name_node.children[0].value
            else: 
                var_name = "Unnamed " + (str(unnamed_index) if unnamed_index > 0 else "")
                unnamed_index += 1
            min_pos = min(token.start_pos for token in var_exp.children[-1].scan_values(lambda x: isinstance(x, lark.Token)))
            max_pos = max(token.end_pos for token in var_exp.children[-1].scan_values(lambda x: isinstance(x, lark.Token)))
            print(var_name, query[min_pos:max_pos])
            parsed_variables[var_name] = {"query": query[min_pos:max_pos], "enabled": True}
        return jsonify({"success": True, "variables": parsed_variables})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)})
    


