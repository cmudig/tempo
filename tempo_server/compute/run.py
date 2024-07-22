"""
Module that manages all of the 
"""
from functools import partial
from .worker import BackgroundWorker
from .filesystem import LocalFilesystem
from .dataset import Dataset
from .model import Model
from .slicefinder import SliceFinder
from .utils import Commands, make_query_result_summary
from divisi.utils import convert_to_native_types

# these variables are only used in the worker
cache_dataset = None # tuple (name, Dataset)
cache_worker_sample_dataset = None # tuple (name, Dataset)

def _get_dataset(filesystem, dataset_name):
    """Only used by the worker to get a full-size dataset"""
    global cache_dataset
    if cache_dataset is None or cache_dataset[1] != dataset_name:
        ds = Dataset(filesystem.subdirectory("datasets", dataset_name))
        ds.load_data()
        cache_dataset = (dataset_name, ds)
    return cache_dataset[1]

def _get_worker_sample_dataset(filesystem, dataset_name):
    """Only used by the worker to get a full-size dataset"""
    global cache_worker_sample_dataset
    if cache_worker_sample_dataset is None or cache_worker_sample_dataset[1] != dataset_name:
        ds = Dataset(filesystem.subdirectory("datasets", dataset_name))
        ds.load_data()
        cache_worker_sample_dataset = (dataset_name, ds)
    return cache_worker_sample_dataset[1]

def task_runner(filesystem, task_info, update_fn):
    print(f"My filesystem is {filesystem}, performing task {task_info}")
    cmd = task_info['cmd']
    if cmd == Commands.TRAIN_MODEL:
        update_fn({'message': 'Loading data'})
        dataset = _get_dataset(filesystem, task_info['dataset_name'])
            
        model_name = task_info['model_name']
        spec = task_info['spec']
        update_fn({'message': 'Loading variables'})
        
        # Train model in a temp directory first
        dest_path = dataset.model_spec_dir(model_name)
        dest_cache_path = dataset.model_cache_dir(model_name)
        try:
            model = Model(filesystem.make_temporary_directory())
            model.make_model(dataset, spec, update_fn=update_fn)
        except Exception as e:
            # Save model spec with an error associated with it
            error_model = Model(dest_path)
            error_model.write_spec({**spec, "error": str(e)})
            raise e
        
        # Transfer model to the target directory
        if dest_path.exists(): dest_path.delete()
        if dest_cache_path.exists(): dest_cache_path.delete()
        model = model.copy_to(dest_path, dest_cache_path)
        
        # Create a default slicing variable spec for the model
        update_fn({'message': 'Creating slicing variables'})
        slicing_spec_name = dataset.default_slicing_variable_spec_name(model_name)
        slicing_spec = dataset.get_slicing_variable_spec(slicing_spec_name)
        variables_df = model.make_modeling_variables(dataset.make_query_engine(), spec, dummies=False)
        slicing_spec.create_default(spec, variables_df)
        
        slicefinder = SliceFinder(dataset)
        slicefinder.invalidate_model(model_name)
        slicefinder.invalidate_variable_spec(slicing_spec_name)
        
    elif cmd == Commands.SUMMARIZE_DATASET:        
        update_fn({'message': 'Loading data'})
        dataset = _get_worker_sample_dataset(filesystem, task_info['dataset_name'])
        
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
        update_fn({'message': 'Loading data'})
        dataset = _get_dataset(filesystem, task_info['dataset_name'])
        
        if isinstance(task_info['query'], dict):
            path = dataset.generate_downloadable_batch_queries(task_info['query'], update_fn=update_fn)
        else:
            path = dataset.generate_downloadable_query(task_info['query'], update_fn=update_fn)
        return path
    elif cmd == Commands.FIND_SLICES:
        update_fn({'message': 'Loading data'})
        dataset = _get_dataset(filesystem, task_info['dataset_name'])
        print("Split sizes:", [len(x) for x in dataset.split_ids])
        
        slicefinder = SliceFinder(dataset)
        slicefinder.find_slices(task_info['model_name'], 
                                task_info['variable_spec_name'], 
                                task_info['score_function_spec'], 
                                update_fn=update_fn,
                                ignore_cache=True,
                                **task_info['options'])
        
        
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