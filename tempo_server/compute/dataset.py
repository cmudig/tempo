import pandas as pd
from tempo_server.query_language.data_types import *
from tempo_server.query_language.evaluator import TrajectoryDataset, get_all_trajectory_ids
from sklearn.model_selection import train_test_split

DEFAULT_SPLIT = {"train": 0.5, "val": 0.25, "test": 0.25}

class Dataset:
    def __init__(self, dataset_fs, split=None):
        """
        dataset_fs: A Filesystem object that stores the dataset.
        """
        self.fs = dataset_fs
        self.spec = self.fs.read_file("spec.json")
        self.split = split
        
        self.data_fs = self.fs.subdirectory("data")
        self.model_fs = self.fs.subdirectory("models")
        self.slices_fs = self.fs.subdirectory("slices")
        
        self.global_cache_dir = self.fs.subdirectory("_cache")
        if split is not None:
            self.split_cache_dir = self.fs.subdirectory(f"_cache_{split}")
        else:
            self.split_cache_dir = self.global_cache_dir
            
        self.dataset = None
        
        self.id_uniques = None
        self.split_ids = None # tuple (train, val, test) ids
        
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
        
    def load_data(self):
        data_config = self.spec.get("data", {})
        attrib_config = data_config.get("attributes", {})
        if "path" in attrib_config:
            df = self.data_fs.read_file(attrib_config["path"])
            id_field = attrib_config.get("id_field", "id")
            if id_field in df.columns:
                df = df.set_index(id_field, drop=True)
            else:
                print("ID column not found in attributes, using the dataframe index")
                df = df.set_index(self.assign_numerical_ids(df.index))
            attributes = AttributeSet(df)
        else:
            attributes = None
        event_config = data_config.get("events", {})
        if "path" in event_config:
            df = self.data_fs.read_file(event_config["path"])
            id_field = event_config.get("id_field", "id")
            df = df.assign(**{id_field: self.assign_numerical_ids(df[id_field])})
            events = EventSet(df,
                            id_field=id_field, 
                            type_field=event_config.get("type_field", "eventtype"), 
                            value_field=event_config.get("value_field", "value"), 
                            time_field=event_config.get("time_field", "time"))
        else:
            events = None
        interval_config = data_config.get("intervals", {})
        if "path" in interval_config:
            df = self.data_fs.read_file(interval_config["path"])
            id_field = interval_config.get("id_field", "id")
            # prev_unique = len(df[id_field].unique())
            df = df.assign(**{id_field: self.assign_numerical_ids(df[id_field])})
            # assert prev_unique == len(df[id_field].unique()), "NOT SAME NUMBER OF IDS"
            intervals = IntervalSet(df,
                                    id_field=id_field, 
                                    type_field=interval_config.get("type_field", "intervaltype"), 
                                    value_field=interval_config.get("value_field", "value"), 
                                    start_time_field=interval_config.get("start_time_field", "starttime"),
                                    end_time_field=interval_config.get("end_time_field", "endtime"))
        else:
            intervals = None
            
        assert attributes is not None or events is not None or intervals is not None, "At least one of attributes, events, or intervals must be provided"
        ids = get_all_trajectory_ids(attributes, events, intervals)
            
        valid_ids = ids
        if attributes is not None:
            attributes = attributes.filter(attributes.get_ids().isin(valid_ids))
        if events is not None:
            events = events.filter(events.get_ids().isin(valid_ids))
        if intervals is not None:
            intervals = intervals.filter(intervals.get_ids().isin(valid_ids))
        
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
            if attributes is not None:
                attributes = attributes.filter(attributes.get_ids().isin(split_ids))
            if events is not None:
                events = events.filter(events.get_ids().isin(split_ids))
            if intervals is not None:
                intervals = intervals.filter(intervals.get_ids().isin(split_ids))

        if "macros" in data_config:
            macros = self.data_fs.read_file(data_config["macros"]["path"]
                                            if "path" in data_config["macros"] 
                                            else "macros.json")
        else:
            macros = {}
            
        self.dataset = TrajectoryDataset(attributes, 
                                    events, 
                                    intervals, 
                                    cache_fs=self.split_cache_dir, 
                                    eventtype_macros=macros)
        self.split_ids = (train_ids, val_ids, test_ids)

    def query(self, query_string, variable_transform=None, use_cache=True, update_fn=None):
        assert self.dataset is not None, "Need to load dataset before querying"
        return self.dataset.query(query_string, variable_transform=variable_transform, use_cache=use_cache, update_fn=update_fn)
    
    def parse(self, query, keep_all_tokens=False):
        assert self.dataset is not None, "Need to load dataset before parsing"
        return self.dataset.parse(query, keep_all_tokens=keep_all_tokens)
    