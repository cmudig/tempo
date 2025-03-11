"""
Module that manages all of the 
"""
from functools import partial
from .worker import BackgroundWorker
from .filesystem import LocalFilesystem, GCSFilesystem
from .dataset import Dataset
from .model import Model
from .slicefinder import SliceFinder
from .utils import Commands, make_query_result_summary
from divisi.utils import convert_to_native_types
import logging
import traceback
from flask_login import current_user

# these variables are only used in the worker
cache_dataset = None # tuple (name, Dataset)
cache_worker_sample_dataset = None # tuple (name, Dataset)

def make_filesystem_from_info(fs_info):
    if fs_info['type'] == 'local':
        return LocalFilesystem(fs_info['path'])
    elif fs_info['type'] == 'gcs':
        from google.cloud import storage
        print('gcs info:', fs_info)
        return GCSFilesystem(storage.Client(), 
                             fs_info['bucket'], 
                             fs_info.get('base_path', ''),
                             local_fallback=fs_info.get('local_fallback', None))
    else:
        raise ValueError(f"Unknown filesystem type: {fs_info['type']}")
    
def _get_dataset(filesystem, dataset_name):
    """Only used by the worker to get a full-size dataset"""
    global cache_dataset
    if cache_dataset is None or cache_dataset[0] != dataset_name:
        ds = Dataset(filesystem.subdirectory("datasets", dataset_name))
        ds.load_data()
        cache_dataset = (dataset_name, ds)
    return cache_dataset[1]

def _get_worker_sample_dataset(filesystem, dataset_name):
    """Only used by the worker to get a sample dataset for the test set"""
    global cache_worker_sample_dataset
    if cache_worker_sample_dataset is None or cache_worker_sample_dataset[0] != dataset_name:
        ds = Dataset(filesystem.subdirectory("datasets", dataset_name), "test")
        ds.load_data()
        cache_worker_sample_dataset = (dataset_name, ds)
    return cache_worker_sample_dataset[1]

def task_runner(fs_info, partition_by_user, task_info, update_fn):
    filesystem = make_filesystem_from_info(fs_info)
    if partition_by_user:
        if "user_id" not in task_info:
            raise ValueError("User authentication required, but no user_id provided")
        filesystem = filesystem.subdirectory("users", task_info["user_id"])
    logging.info(f"My filesystem is {filesystem}, performing task {task_info}")
        
    cmd = task_info['cmd']
    if cmd == Commands.BUILD_DATASET:
        print("BUILD DATASET")
        update_fn({'message': 'Building dataset'})
        dataset_name = task_info['dataset_name']
        
        spec = task_info["spec"]
        if "error" in spec: del spec["error"]
        
        # Make sure the dataset can be loaded in the draft stage
        update_fn({'message': 'Sandboxing dataset for validation'})
        sandbox_dataset = Dataset(filesystem.subdirectory("dataset_sandbox", dataset_name))
        if sandbox_dataset.fs.exists():
            sandbox_dataset.fs.delete()
        filesystem.subdirectory("datasets", dataset_name).copy_directory_contents(sandbox_dataset.fs)
        if filesystem.subdirectory("dataset_drafts", dataset_name).exists():
            filesystem.subdirectory("dataset_drafts", dataset_name).copy_directory_contents(sandbox_dataset.fs)
        sandbox_dataset.write_spec(spec)
        
        try:
            update_fn({'message': 'Loading data for validation'})
            sandbox_dataset.load_data()
            
            update_fn({'message': 'Transferring files'})
            
            # Delete existing data files
            old_dataset = Dataset(filesystem.subdirectory("datasets", dataset_name))
            old_spec = old_dataset.get_spec()
            for source in old_spec.get("data", {}).get("sources", []):
                if source["path"]:
                    old_dataset.fs.delete(source["path"])
            old_dataset.global_cache_dir.delete()
            if isinstance(old_dataset.global_cache_dir, GCSFilesystem) and old_dataset.global_cache_dir.get_local_fallback():
                old_dataset.global_cache_dir.get_local_fallback().delete()
                    
            # Move new data files
            old_dataset.write_spec(spec)
            for source in spec.get("data", {}).get("sources", []):
                if source["path"]:
                    sandbox_dataset.fs.copy_file(old_dataset.fs, source["path"])
                    
            # Delete draft and sandbox
            sandbox_dataset.fs.delete()
            filesystem.subdirectory("dataset_drafts", dataset_name).delete()
            
            update_fn({'message': 'Loading dataset'})
            new_dataset = Dataset(filesystem.subdirectory("datasets", dataset_name))
            new_dataset.load_data()
        except Exception as e:
            logging.info("Error building dataset: " + traceback.format_exc())
            Dataset(filesystem.subdirectory("dataset_drafts", dataset_name)).write_spec({**spec, "error": str(e)})
            raise e
            
        del sandbox_dataset
        print("DONE BUILDING DATASET")
    elif cmd == Commands.TRAIN_MODEL:
        update_fn({'message': 'Loading data'})
        dataset = _get_dataset(filesystem, task_info['dataset_name'])
        logging.info(f"Split sizes: {[len(x) for x in dataset.split_ids]}")
                
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
            logging.info("Error training model: " + traceback.format_exc())
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
        variables_df = model.make_modeling_variables(dataset.make_query_engine(), 
                                                     spec, 
                                                     update_fn=lambda m: update_fn({'message': 'Creating slicing variables: ' + m['message']}), 
                                                     dummies=False)
        slicing_spec.create_default(spec, variables_df)
        
        slicefinder = SliceFinder(dataset)
        slicefinder.invalidate_model(model_name)
        slicefinder.invalidate_variable_spec(slicing_spec_name)
        
    elif cmd == Commands.SUMMARIZE_DATASET:        
        update_fn({'message': 'Loading data'})
        dataset = _get_worker_sample_dataset(filesystem, task_info['dataset_name'])
        
        result = {}
        query_engine = dataset.make_query_engine()
        update_fn({'message': 'Summarizing attributes'})
        result["attributes"] = {attr_name: make_query_result_summary(query_engine, attr_set.get(attr_name))
                                for attr_set in dataset.attributes
                                for attr_name in attr_set.df.columns}
        update_fn({'message': 'Summarizing events'})
        result["events"] = {eventtype: make_query_result_summary(query_engine, event_set.get(eventtype))
                            for event_set in dataset.events
                            for eventtype in event_set.get_types().unique()}
        update_fn({'message': 'Summarizing intervals'})
        result["intervals"] = {eventtype: make_query_result_summary(query_engine, interval_set.get(eventtype))
                               for interval_set in dataset.intervals
                            for eventtype in interval_set.get_types().unique()}
        update_fn({'message': 'Generating report'})
        data_summary = convert_to_native_types(result)
        dataset.split_cache_dir.write_file(data_summary, "summary.json")
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
        logging.info(f"Split sizes: {[len(x) for x in dataset.split_ids]}")
        
        slicefinder = SliceFinder(dataset)
        slicefinder.find_slices(task_info['model_name'], 
                                task_info['variable_spec_name'], 
                                task_info['score_function_spec'], 
                                update_fn=update_fn,
                                ignore_cache=True,
                                **task_info['options'])
    elif cmd == Commands.RUN_MODEL_INFERENCE:
        update_fn({'message': 'Loading data'})
        dataset = _get_dataset(filesystem, task_info['dataset_name'])
                
        model_name = task_info['model_name']
        model = dataset.get_model(model_name)
        model.compute_model_predictions(dataset, ids=task_info.get('ids'), inputs=task_info.get('inputs'), update_fn=update_fn)

    elif cmd == Commands.GET_MODEL_INSTANCES:
        update_fn({'message': 'Loading data'})
        dataset = _get_dataset(filesystem, task_info['dataset_name'])
                
        model_name = task_info['model_name']
        model = dataset.get_model(model_name)
        modeling_df, outcome, index = model.get_modeling_inputs(dataset, task_info['ids'], update_fn=update_fn)
        return convert_to_native_types([
            {"index": index[i],
             "inputs": modeling_df.iloc[i].to_dict(),
             "ground_truth": outcome[i]
             } for i in range(len(index))
        ])
    elif cmd == Commands.CLEAR_MEMORY_CACHE:
        global cache_dataset, cache_worker_sample_dataset
        cache_dataset = None
        cache_worker_sample_dataset = None
        
    return "Success"
    
# these variables are only used in the main script
worker = None
filesystem = None
demo_data_fs = None
sample_dataset = None # tuple (name, Dataset)
fs_partition_by_user = False

def setup_worker(fs_info, log_path, verbose=False, partition_by_user=False):
    global worker
    global filesystem
    global fs_partition_by_user
    global demo_data_fs
    fs_partition_by_user = partition_by_user
    worker = BackgroundWorker(partial(task_runner, fs_info, partition_by_user), log_path, verbose=verbose, partition_by_user=partition_by_user)
    filesystem = make_filesystem_from_info(fs_info)
    
    if 'demo_data' in fs_info:
        demo_data_fs = filesystem.subdirectory(fs_info['demo_data'])
    else:
        demo_data_fs = None

    return worker

def get_worker():
    return worker

def get_filesystem():
    if fs_partition_by_user and current_user.is_authenticated:
        return filesystem.subdirectory("users", current_user.get_id())
    return filesystem

def get_demo_data_fs():
    return demo_data_fs

def get_sample_dataset(dataset_name):
    global sample_dataset
    if sample_dataset is None or sample_dataset[0] != dataset_name:
        ds = Dataset(get_filesystem().subdirectory("datasets", dataset_name), split="test")
        ds.load_data()
        sample_dataset = (dataset_name, ds)
        
    return sample_dataset[1]

def clear_sample_dataset():
    global sample_dataset
    sample_dataset = None