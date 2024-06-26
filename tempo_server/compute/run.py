"""
Module that manages all of the 
"""
from functools import partial
from .worker import BackgroundWorker
from .filesystem import LocalFilesystem
from .dataset import Dataset
from .model import Model
from .utils import Commands, make_query_result_summary
from divisi.utils import convert_to_native_types

# these variables are only used in the worker
cache_dataset = None # tuple (name, Dataset)
cache_worker_sample_dataset = None # tuple (name, Dataset)

def task_runner(filesystem, task_info, update_fn):
    global cache_dataset, cache_worker_sample_dataset
    
    print(f"My filesystem is {filesystem}, performing task {task_info}")
    cmd = task_info['cmd']
    if cmd == Commands.TRAIN_MODEL:
        if cache_dataset is None or cache_dataset[1] != task_info['dataset_name']:
            update_fn({'message': 'Loading data'})
            ds = Dataset(filesystem.subdirectory("datasets", task_info['dataset_name']))
            ds.load_data()
            cache_dataset = (task_info['dataset_name'], ds)
        dataset = cache_dataset[1]
            
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
    elif cmd == Commands.SUMMARIZE_DATASET:        
        if cache_worker_sample_dataset is None or cache_worker_sample_dataset[1] != task_info['dataset_name']:
            update_fn({'message': 'Loading data'})
            ds = Dataset(filesystem.subdirectory("datasets", task_info['dataset_name']), split="test")
            ds.load_data()
            cache_worker_sample_dataset = (task_info['dataset_name'], ds)    
        dataset = cache_worker_sample_dataset[1].dataset
        
        result = {}
        update_fn({'message': 'Summarizing attributes'})
        result["attributes"] = {attr_name: make_query_result_summary(dataset, dataset.attributes.get(attr_name))
                                for attr_name in dataset.attributes.df.columns}
        update_fn({'message': 'Summarizing events'})
        result["events"] = {eventtype: make_query_result_summary(dataset, dataset.events.get(eventtype))
                            for eventtype in dataset.events.get_types().unique()}
        update_fn({'message': 'Summarizing intervals'})
        result["intervals"] = {eventtype: make_query_result_summary(dataset, dataset.intervals.get(eventtype))
                            for eventtype in dataset.intervals.get_types().unique()}
        update_fn({'message': 'Generating report'})
        data_summary = convert_to_native_types(result)
        cache_worker_sample_dataset[1].split_cache_dir.write_file(data_summary, "summary.json")
    elif cmd == Commands.GENERATE_QUERY_DOWNLOAD:
        if cache_dataset is None or cache_dataset[1] != task_info['dataset_name']:
            update_fn({'message': 'Loading data'})
            ds = Dataset(filesystem.subdirectory("datasets", task_info['dataset_name']))
            ds.load_data()
            cache_dataset = (task_info['dataset_name'], ds)
        dataset = cache_dataset[1]
        
        if isinstance(task_info['query'], dict):
            path = dataset.generate_downloadable_batch_queries(task_info['query'], update_fn=update_fn)
        else:
            path = dataset.generate_downloadable_query(task_info['query'], update_fn=update_fn)
        return path
        
    return "Success"
    
# these variables are only used in the main script
worker = None
filesystem = None
sample_dataset = None # tuple (name, Dataset)

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

def get_sample_dataset(dataset_name):
    global sample_dataset
    if sample_dataset is None or sample_dataset[0] != dataset_name:
        ds = Dataset(filesystem.subdirectory("datasets", dataset_name), split="test")
        ds.load_data()
        sample_dataset = (dataset_name, ds)
        
    return sample_dataset[1]