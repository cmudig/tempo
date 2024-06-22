"""
Module that manages all of the 
"""
from functools import partial
from .worker import BackgroundWorker
from .utils import Commands

def task_runner(filesystem, task_info, update_fn):
    print(f"My filesystem is {filesystem}, performing task {task_info}")
    return "Success"
    
worker = None

def setup_worker(filesystem):
    global worker
    worker = BackgroundWorker(partial(task_runner, filesystem))
    return worker

def get_worker():
    return worker