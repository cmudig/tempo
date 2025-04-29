"""
A reusable module for running a lightweight multiprocessing background worker 
with cancelable tasks and status reporting.
"""
import multiprocessing
import time
import os
import threading
import _thread
import uuid
import traceback
import logging
import logging.handlers
from flask_login import current_user

from functools import partial

mp = multiprocessing.get_context('spawn')

def listener_process(queue, path):
    if path is not None:
        root = logging.getLogger()
        h = logging.handlers.RotatingFileHandler(os.path.join(path, 'log.txt'), 'a', 100000, 10)
        f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
        h.setFormatter(f)
        root.addHandler(h)
    
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            
class TaskStatus:
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELING = "canceling"
    CANCELED = "canceled"

def interrupt_handler(interrupt_event):
    while True:
        interrupt_event.wait()
        print("[background_worker_interrupt] Interrupting")
        _thread.interrupt_main()
        interrupt_event.clear()

def configure_worker_logging(queue):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)
    
def start_worker(task_runner, interrupt_event, request_queue, status, log_queue, verbose=False):
    """
    Args:
    * task_runner: A function that takes a task_info object and an update
        function and performs the task. The update function takes a single info
        argument and stores that as the additional status element in the task
        status. The task runner function may return a value, which will be stored
        as the status element in the completed task status. If the task runner
        raises an Exception, the text of the exception will be stored as the
        status element in the error task status.
    * interrupt_event: A multiprocessing Event that gets set when the currently
        running task should be interrupted.
    * request_queue: A multiprocessing Queue that passes tasks to be run in the
        background. Each task is a tuple of three elements: a task ID string,
        and an object containing task info describing how to perform the task.
    """

    configure_worker_logging(log_queue)
    
    task = threading.Thread(target=interrupt_handler, args=(interrupt_event,))
    task.start()

    if verbose:
        logging.info(f"[background_worker] Starting PID: {os.getpid()}")
    while True:
        task_id, task_info = request_queue.get()
        try:
            if task_id in status and status[task_id][0] == TaskStatus.CANCELING:
                if verbose:
                    logging.info(f"[background_worker] Aborting task {task_id}")
                status[task_id] = (TaskStatus.CANCELED, None)
                continue
            
            status[task_id] = (TaskStatus.RUNNING, None)
            if verbose:
                logging.info(f"[background_worker] Running command: {task_id}")
            def update_fn(update_info):
                status[task_id] = (TaskStatus.RUNNING, update_info)
            try:
                result = task_runner(task_info, update_fn)
            except KeyboardInterrupt:
                if verbose:
                    logging.info("[background_worker] Received interrupt")
                status[task_id] = (TaskStatus.CANCELED, None)
            except Exception as e:
                if verbose:
                    logging.info("[background_worker] Exception raised")
                    logging.info(traceback.format_exc())
                status[task_id] = (TaskStatus.ERROR, str(e))
            else:
                status[task_id] = (TaskStatus.COMPLETE, result)
        except KeyboardInterrupt:
            if verbose:
                logging.info("[background_worker] Received interrupt")
            status[task_id] = (TaskStatus.CANCELED, None)
            
class BackgroundWorker:
    def __init__(self, task_runner, log_path, verbose=False, partition_by_user=False):
        """
        Args:
        * task_runner: A function that takes a task_info object and an update
        function and performs the task. The update function takes a single info
        argument and stores that as the additional status element in the task
        status. The task runner function may return a value, which will be stored
        as the status element in the completed task status. If the task runner
        raises an Exception, the text of the exception will be stored as the
        status element in the error task status. NOTE: This function must be in
        the global scope for the multiprocessing implementation to work correctly.
        * verbose: Whether or not to output background worker messages.
        
        If partition_by_user is True, all tasks will be filtered and submitted
        with user ID information.
        """
        self.task_runner = task_runner
        self.verbose = verbose
        self.interrupt_event = mp.Event()
        self.request_queue = mp.Queue()
        self.manager = mp.Manager()
        self.task_status = self.manager.dict()
        self.task_cache = {}
        self.task_lock = mp.Lock()
        self.worker_process = None
        self.partition_by_user = partition_by_user
        
        self.log_queue = mp.Queue()
        self.log_path = log_path
        
    def start(self):
        assert self.worker_process is None, "Cannot start a worker process while an existing worker is running"
        self.worker_process = mp.Process(target=partial(start_worker, self.task_runner, verbose=self.verbose), 
                                         args=(self.interrupt_event, 
                                               self.request_queue, 
                                               self.task_status,
                                               self.log_queue))
        self.worker_process.start()
        self.log_process = mp.Process(target=listener_process,
                                       args=(self.log_queue, self.log_path))
        self.log_process.start()
        
    def submit_task(self, task_info):
        if self.partition_by_user and current_user.is_authenticated:
            task_info = {**task_info, "user_id": current_user.get_id()}
        with self.task_lock:
            task_id = uuid.uuid4().hex
            task_summary = (task_id, task_info)
            self.task_cache[task_id] = task_info
            self.task_status[task_id] = (TaskStatus.WAITING, None)
            self.request_queue.put(task_summary)
            return task_id
    
    def status(self, task_id):
        if self.partition_by_user and current_user.is_authenticated and task_id in self.task_status:
            if self.task_cache[task_id]["user_id"] != current_user.get_id():
                raise KeyError(task_id)
        return self.task_status[task_id]
    
    def task_info(self, task_id):
        if self.partition_by_user and current_user.is_authenticated and task_id in self.task_status:
            if self.task_cache[task_id]["user_id"] != current_user.get_id():
                raise KeyError(task_id)
        status = self.task_status[task_id]
        return {
            'id': task_id, 
            'info': self.task_cache[task_id], 
            'status': status[0],
            **({'status_info': status[1]} if status[0] is not None else {})
        }
        
    def current_jobs(self):
        return [{
            'id': id, 
            'info': self.task_cache[id], 
            'status': status[0],
            **({'status_info': status[1]} if status[1] is not None else {})
        } for id, status in self.task_status.items() 
                if status[0] in (TaskStatus.WAITING, TaskStatus.RUNNING)
                and not (self.partition_by_user and current_user.is_authenticated and
                         self.task_cache[id]['user_id'] != current_user.get_id())]
        
    def all_jobs(self):
        return [{
            'id': id, 
            'info': self.task_cache[id], 
            'status': status[0],
            **({'status_info': status[1]} if status[1] is not None else {})
        } for id, status in self.task_status.items()
                if not (self.partition_by_user and current_user.is_authenticated and
                         self.task_cache[id]['user_id'] != current_user.get_id())]
        
    def cancel_task(self, task_id):
        with self.task_lock:
            if self.partition_by_user and current_user.is_authenticated and task_id in self.task_status:
                if self.task_cache[task_id]["user_id"] != current_user.get_id():
                    logging.info(f"[cancel_task]: Task {task_id} not in list of tasks for this user")
                    return
            if task_id not in self.task_status:
                logging.info(f"[cancel_task]: Task {task_id} not in list of tasks")
                return
            current_status = self.task_status[task_id][0]
            logging.info(f"[cancel_task]: Task {task_id} status: {current_status}")
            if current_status == TaskStatus.RUNNING:
                self.interrupt_event.set()
            elif current_status == TaskStatus.WAITING:
                self.task_status[task_id] = (TaskStatus.CANCELING, None)
        
    def terminate(self):
        if self.worker_process is not None:
            self.worker_process.terminate()
            self.worker_process = None
        if self.request_queue is not None:
            self.request_queue.close()
            self.request_queue = None
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None
        if self.log_process is not None:
            self.log_queue.put_nowait(None)
            self.log_process.terminate()
            self.log_queue = None
            self.log_process = None            
            
def example_runner(info, update_fn):
    if info == "my_task":
        for i in range(5):
            update_fn({"progress": i / 5 * 100})
            time.sleep(1)
        return "Success!"
    else:
        time.sleep(1)
        raise ValueError("Don't like this task")

if __name__ == '__main__':
    try:
        worker = BackgroundWorker(example_runner, verbose=True)
        worker.start()
        id = worker.submit_task("my_task")
        id2 = worker.submit_task("my_other_task")
        id3 = worker.submit_task("a_third_task")
        print("task id:", id)
        while True:
            time.sleep(1)
            if worker.status(id3)[0] == TaskStatus.WAITING:
                worker.cancel_task(id3)
            print(worker.all_jobs())
            if not worker.current_jobs():
                break
        print("Terminating")
        worker.terminate()
    except KeyboardInterrupt:
        worker.terminate()
        print("Shut down background worker")