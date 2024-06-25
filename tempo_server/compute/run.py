"""
Module that manages all of the 
"""
from functools import partial
from .worker import BackgroundWorker
from .filesystem import LocalFilesystem
from .dataset import Dataset
from .model import Model
from .utils import Commands

dataset = None

def task_runner(filesystem, task_info, update_fn):
    global dataset

    print(f"My filesystem is {filesystem}, performing task {task_info}")
    cmd = task_info['cmd']
    if cmd == Commands.TRAIN_MODEL:
        if dataset is None:
            update_fn({'message': 'Loading data'})
            dataset = Dataset(filesystem.subdirectory("datasets", task_info['dataset_name']))
            dataset.load_data()
            
        model_name = task_info['model_name']
        spec = task_info['spec']
        update_fn({'message': 'Loading variables'})
        
        # Train model in a temp directory first
        dest_path = dataset.fs.subdirectory("models", model_name)
        try:
            model = Model(filesystem.make_temporary_directory())
            model.make_model(dataset, spec, update_fn=update_fn)
        except Exception as e:
            # Save model spec with an error associated with it
            error_model = Model(dest_path)
            error_model.write_spec({**spec, "error": str(e)})
            raise e
        else:
            # Transfer model to the target directory
            if dest_path.exists(): dest_path.delete()
            model.fs.copy_directory_contents(dest_path)

    return "Success"
    
worker = None
filesystem = None

def setup_worker(fs, verbose=False):
    global worker
    global filesystem
    worker = BackgroundWorker(partial(task_runner, fs), verbose=verbose)
    filesystem = fs
    return worker

def get_worker():
    return worker

def get_filesystem():
    return filesystem