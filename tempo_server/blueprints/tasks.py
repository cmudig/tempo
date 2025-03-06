from flask import Blueprint, jsonify, request
from flask_login import login_required
from ..compute.run import get_worker
from ..compute.worker import TaskStatus

tasks_blueprint = Blueprint('tasks', __name__)

# Getting tasks and their status

@tasks_blueprint.get("/tasks")
@login_required
def list_tasks():
    """
    Query parameters: keys and values for filtering the results list by the task
        info;
        * all: if 1, return all tasks including completed ones, otherwise 0 
            (default)

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
    jobs = get_worker().all_jobs() if request.args.get("all", "0") == "1" else get_worker().current_jobs()
    for k, v in request.args.items():
        if k == "all": continue
        jobs = [job for job in jobs if k in job['info'] and job['info'][k] == v]
    return jsonify(jobs)
    
@tasks_blueprint.get("/tasks/<task_id>")
@login_required
def task_status(task_id):
    """
    Returns: JSON of the format { 
        "id": task id,
        "info": task info,
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
    
@tasks_blueprint.post("/tasks/<task_id>/stop")
@login_required
def stop_task(task_id):
    """
    Parameters:
    * task_id: the ID of the task to stop
    
    Returns: plain text message if the task was stopped successfully
    """
    worker = get_worker()
    try:
        status = worker.status(task_id)
        if status[0] not in (TaskStatus.WAITING, TaskStatus.RUNNING):
            return "Task is not pending", 400
    except:
        return "Task does not exist", 404
    else:
        worker.cancel_task(task_id)
        return "Success"
    
