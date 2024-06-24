from flask import Blueprint, jsonify
from ..compute.run import get_worker

tasks_blueprint = Blueprint('tasks', __name__)

# Getting tasks and their status

@tasks_blueprint.get("/tasks")
def list_tasks():
    """
    Returns: JSON array of the format [
        { 
            "id": "task id", 
            "info": task info, 
            "status": "status string",
            "status_info": status info 
        },
        ...
    ]
    """
    worker = get_worker()
    return jsonify(worker.current_jobs())
    
@tasks_blueprint.get("/tasks/<task_id>")
def task_status(task_id):
    """
    Returns: JSON of the format { 
        "status": "status string",
        "status_info": status info 
    }
    """
    worker = get_worker()
    try:
        status = worker.status(task_id)
    except:
        return "Task does not exist", 404
    else:
        return jsonify({ "status": status[0], "status_info": status[1] })
    
