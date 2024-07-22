import pandas as pd
from tempo_server.query_language.data_types import *
from tempo_server.query_language.evaluator import QueryEngine, get_all_trajectory_ids
from sklearn.model_selection import train_test_split
import zipfile
from io import BytesIO
import time
import uuid
from tempo_server.compute.model import Model
from tempo_server.compute.slicefinder import SlicingVariableSpec

DEFAULT_SPLIT = {"train": 0.5, "val": 0.25, "test": 0.25}

class Dataset:
    def __init__(self, dataset_fs, split=None):
        """
        dataset_fs: A Filesystem object that stores the dataset.
        """
        self.fs = dataset_fs
        self.spec = self.fs.read_file("spec.json")
        self.split = split
        
        self.global_cache_dir = self.fs.subdirectory("_cache")
        if split is not None:
            self.split_cache_dir = self.global_cache_dir.subdirectory(f"_{split}")
        else:
            self.split_cache_dir = self.global_cache_dir.subdirectory(f"_all")
            
        self.dataset = None
        
        self.id_uniques = None
        self.attributes = None
        self.events = None
        self.intervals = None
        self.macros = None
        self.split_ids = None # tuple (train, val, test) ids
        
    def model_cache_dir(self, model_name):
        return self.global_cache_dir.subdirectory("models", model_name)
    
    def model_spec_dir(self, model_name):
        return self.fs.subdirectory("models", model_name)        

    def get_spec(self):
        return self.spec
    
    def write_spec(self, new_spec):
        self.spec = new_spec
        self.fs.write_file(new_spec, "spec.json")
        
    def get_models(self):
        """Returns a dictionary of model name to Model objects."""
        model_dir = self.fs.subdirectory("models")
        return {model_name: self.get_model(model_name)
                for model_name in model_dir.list_files()
                if self.model_spec_dir(model_name).exists("spec.json")}
    
    def get_model(self, model_name):
        return Model(self.model_spec_dir(model_name), self.model_cache_dir(model_name))
    
    def default_slicing_variable_spec_name(self, model_name):
        return f"{model_name} (Default)"
        
    def get_slicing_variable_specs(self):
        spec_dir = self.fs.subdirectory("slicing_specs")
        return {spec_name: self.get_slicing_variable_spec(spec_name)
                for spec_name in spec_dir.list_files()
                if spec_dir.exists(spec_name, "spec.json")}
        
    def get_slicing_variable_spec(self, spec_name):
        return SlicingVariableSpec(self.fs.subdirectory("slicing_specs", spec_name))
        
    def assign_numerical_ids(self, ids):
        """
        Converts IDs to integers if they are not already.
        """
        if pd.api.types.is_integer_dtype(ids.dtype):
            assert self.id_uniques is None, "IDs have different types in different source dataframes"
            return ids
        else:
            if self.id_uniques is None:
                categorical_rep = ids.astype('category')
                self.id_uniques = categorical_rep.cat.categories
                return categorical_rep.cat.codes
            categorical_rep = ids.astype('category')
            new_vals = categorical_rep.cat.categories[~categorical_rep.cat.categories.isin(self.id_uniques)]
            self.id_uniques = self.id_uniques.union(new_vals, sort=False)
            return pd.Categorical(ids, self.id_uniques).codes
        
    def load_if_needed(self):
        if self.attributes is None: self.load_data()
        return self
    
    def load_data(self):
        data_config = self.spec.get("data", {})
        attributes = []
        events = []
        intervals = []
        for source_config in data_config.get("sources", []):
            if "type" not in source_config: raise ValueError(f"source config must have a 'type' field: {source_config}")
            source_type = source_config["type"].lower()
            if source_type not in ("attributes", "events", "intervals"):
                raise ValueError("source type should be one of (attributes, events, intervals)")

            if "path" in source_config:
                df = self.fs.read_file(source_config["path"])
            else:
                print(f"Don't know how to import source {source_config}, skipping")
            
            if source_type == "attributes":
                id_field = source_config.get("id_field", "id")
                if id_field in df.columns:
                    df = df.set_index(id_field, drop=True)
                else:
                    print("ID column not found in attributes, using the dataframe index")
                    df = df.set_index(self.assign_numerical_ids(df.index))
                attributes.append(AttributeSet(df))
                    
            elif source_type == "events":
                id_field = source_config.get("id_field", "id")
                df = df.assign(**{id_field: self.assign_numerical_ids(df[id_field])})
                events.append(EventSet(df,
                                       id_field=id_field, 
                                       type_field=source_config.get("type_field", "eventtype"), 
                                       value_field=source_config.get("value_field", "value"), 
                                       time_field=source_config.get("time_field", "time")))

            elif source_type == "intervals":
                id_field = source_config.get("id_field", "id")
                # prev_unique = len(df[id_field].unique())
                df = df.assign(**{id_field: self.assign_numerical_ids(df[id_field])})
                # assert prev_unique == len(df[id_field].unique()), "NOT SAME NUMBER OF IDS"
                intervals.append(IntervalSet(df,
                                             id_field=id_field, 
                                             type_field=source_config.get("type_field", "intervaltype"), 
                                             value_field=source_config.get("value_field", "value"), 
                                             start_time_field=source_config.get("start_time_field", "starttime"),
                                             end_time_field=source_config.get("end_time_field", "endtime")))
                
        assert attributes or events or intervals, "At least one of attributes, events, or intervals must be provided"
        ids = get_all_trajectory_ids(attributes, events, intervals)
            
        valid_ids = ids
        attributes = [attr_set.filter(attr_set.get_ids().isin(valid_ids))
                      for attr_set in attributes]
        events = [event_set.filter(event_set.get_ids().isin(valid_ids))
                      for event_set in events]
        intervals = [interval_set.filter(interval_set.get_ids().isin(valid_ids))
                      for interval_set in intervals]
        
        if not self.global_cache_dir.exists("train_test_split.pkl"):
            train_split_config = {**data_config.get("split", DEFAULT_SPLIT)}
            assert len(set(train_split_config.keys()) - {"train", "val", "test"}) == 0, "train_split_config must contain only 'train', 'val', and 'test' keys"
            if len(train_split_config) == 3:
                assert abs(sum(train_split_config.values()) - 1.0) <= 1e-3, "Training, val, and test fractions must sum to 1"
            else: 
                while len(train_split_config) < 2:
                    missing_key = next(k for k in DEFAULT_SPLIT if k not in train_split_config)
                    train_split_config[missing_key] = DEFAULT_SPLIT[missing_key]
                missing_key = list({"train", "val", "test"} - set(train_split_config.keys()))[0]
                train_split_config[missing_key] = 1.0 - sum(train_split_config.values())
                
            print(f"Setting up train-val-test split: {train_split_config}")
            train_ids, val_ids = train_test_split(valid_ids, train_size=train_split_config["train"])
            val_ids, test_ids = train_test_split(val_ids, test_size=train_split_config["test"] / (train_split_config["val"] + train_split_config["test"]))
            self.global_cache_dir.write_file((train_ids, val_ids, test_ids), "train_test_split.pkl")
        else:
            train_ids, val_ids, test_ids = self.global_cache_dir.read_file("train_test_split.pkl")

        # print("Original split:", intervals.filter(intervals.get_ids().isin(train_ids)), intervals.filter(intervals.get_ids().isin(val_ids)), intervals.filter(intervals.get_ids().isin(test_ids)))
        if self.split is not None:
            split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[self.split]
            attributes = [attr_set.filter(attr_set.get_ids().isin(split_ids))
                for attr_set in attributes]
            events = [event_set.filter(event_set.get_ids().isin(split_ids))
                        for event_set in events]
            intervals = [interval_set.filter(interval_set.get_ids().isin(split_ids))
                        for interval_set in intervals]

        if "macros" in data_config:
            macros = self.fs.read_file(data_config["macros"]["path"]
                                            if "path" in data_config["macros"] 
                                            else "macros.json")
        else:
            macros = {}
            
        self.attributes = attributes
        self.events = events
        self.intervals = intervals
        self.macros = macros
        self.split_ids = (train_ids, val_ids, test_ids)

    def make_query_engine(self, cache_fs=None):
        """If cache_fs is None, uses the default variable cache."""
        if self.attributes is None:
            self.load_data()
        return QueryEngine(self.attributes, 
                           self.events, 
                           self.intervals, 
                           cache_fs=self.split_cache_dir.subdirectory("variables") if cache_fs is None else cache_fs, 
                           eventtype_macros=self.macros)

    def get_summary(self):
        if not self.split_cache_dir.exists("summary.json"): return None
        return self.split_cache_dir.read_file("summary.json")
    
    def read_downloadable_query_result(self, path):
        return self.global_cache_dir.read_file("query_downloads", path, format='bytes')
    
    def get_downloadable_query(self, query):
        try:
            cache = self.global_cache_dir.read_file("query_downloads", "cache.json")
        except:
            cache = []
        cache_hit = next((item for item in cache if item['query'] == query), None)
        if cache_hit is not None:
            return cache_hit['path']
        return None
    
    def generate_downloadable_query(self, query, update_fn=None):
        """
        Generates a ZIP file containing the result of a given query and saves it
        into the global cache directory. Returns the name of a file in the 
        query_downloads subdirectory of the global cache directory that contains
        the result.
        """
        if (path := self.get_downloadable_query(query)) is not None:
            return path
        
        train_ids, val_ids, test_ids = self.split_ids

        if update_fn is not None: update_fn({'message': 'Running query'})
        engine = self.make_query_engine()
        result = engine.query(query, use_cache=False)

        if update_fn is not None: update_fn({'message': 'Saving results'})
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            data = zipfile.ZipInfo(f'query.txt')
            data.date_time = time.localtime(time.time())[:6]
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, query)
            for filename, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
                data = zipfile.ZipInfo(f'{filename}.csv')
                data.date_time = time.localtime(time.time())[:6]
                data.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(data, result.filter(result.get_ids().isin(ids)).to_csv())
        memory_file.seek(0)
        path = f"{uuid.uuid4().hex}.zip"
        self.global_cache_dir.subdirectory("query_downloads").write_file(memory_file.getvalue(), path, format='bytes')
        
        # Save to cache
        try:
            cache = self.global_cache_dir.read_file("query_downloads", "cache.json")
        except:
            cache = []
        cache.append({ "query": query, "path": path })
        self.global_cache_dir.write_file(cache, "query_downloads", "cache.json")
        return path

    def generate_downloadable_batch_queries(self, queries, update_fn=None):
        """
        Same as generate_downloadable_query but generates one ZIP file
        containing several queries.
        """
        if (path := self.get_downloadable_query(queries)) is not None:
            return path

        engine = self.make_query_engine()        
        train_ids, val_ids, test_ids = self.split_ids

        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            data = zipfile.ZipInfo(f'queries.txt')
            data.date_time = time.localtime(time.time())[:6]
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, "\n\n".join(f"{name}\n{query}" for name, query in queries.items()))
            for i, (query_name, query) in enumerate(queries.items()):
                prog_message = f'Running query {i + 1} of {len(queries)}'
                if update_fn is not None: update_fn({'message': prog_message})
                
                def prog_fn(curr, tot):
                    if update_fn is not None: update_fn({'message': f'{prog_message}: variable {curr} of {tot}'})
                result = engine.query(query, update_fn=prog_fn)
                for filename, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
                    data = zipfile.ZipInfo(f'{query_name}_{filename}.csv')
                    data.date_time = time.localtime(time.time())[:6]
                    data.compress_type = zipfile.ZIP_DEFLATED
                    zf.writestr(data, result.filter(result.get_ids().isin(ids)).to_csv())
        memory_file.seek(0)

        path = f"{uuid.uuid4().hex}.zip"
        self.global_cache_dir.subdirectory("query_downloads").write_file(memory_file.getvalue(), path, format='bytes')
        
        # Save to cache
        try:
            cache = self.global_cache_dir.read_file("query_downloads", "cache.json")
        except:
            cache = []
        cache.append({ "query": queries, "path": path })
        self.global_cache_dir.write_file(cache, "query_downloads", "cache.json")
        return path
