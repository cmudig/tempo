import pandas as pd
import os
import json
import pickle
from query_language.data_types import *
from query_language.evaluator import TrajectoryDataset
from sklearn.model_selection import train_test_split
from model_training import ModelTrainer
from model_slice_finding import SliceDiscoveryHelper, SliceEvaluationHelper

class DatasetManager:
    def __init__(self, base_path):
        """
        base_path: A path to a directory containing a config.json file. 
        """
        self.base_path = base_path
        self.config_path = os.path.join(self.base_path, "config.json")
        assert os.path.exists(self.config_path), f"Dataset base path '{base_path}' must contain a config.json file"
        with open(self.config_path, "r") as file:
            self.config = json.load(file)
        
        self.data_dir = os.path.join(self.base_path, "data")
        assert os.path.exists(self.data_dir) and os.path.isdir(self.data_dir), f"Dataset base path '{base_path}' must contain a data/ directory"
        self.model_dir = os.path.join(self.base_path, "models")
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        self.slices_dir = os.path.join(self.base_path, "slices")
        if not os.path.exists(self.slices_dir): os.mkdir(self.slices_dir)
        
    def load_data(self, sample=False, cache_dir=None, val_only=False):
        data_config = self.config.get("data", {})
        attrib_config = data_config.get("attributes", {})
        if "path" in attrib_config:
            attributes = AttributeSet(pd.read_feather(os.path.join(self.base_path, attrib_config["path"])))
        else:
            attributes = None
        event_config = data_config.get("events", {})
        if "path" in event_config:
            events = EventSet(pd.read_feather(os.path.join(self.base_path, event_config["path"])),
                            id_field=event_config.get("id_field", "id"), 
                            type_field=event_config.get("type_field", "eventtype"), 
                            value_field=event_config.get("value_field", "value"), 
                            time_field=event_config.get("time_field", "time"))
        else:
            events = None
        interval_config = data_config.get("intervals", {})
        if "path" in interval_config:
            intervals = IntervalSet(pd.read_feather(os.path.join(self.base_path, interval_config["path"])),
                                    id_field=event_config.get("id_field", "id"), 
                                    type_field=event_config.get("type_field", "intervaltype"), 
                                    value_field=event_config.get("value_field", "value"), 
                                    start_time_field=event_config.get("start_time_field", "starttime"),
                                    end_time_field=event_config.get("end_time_field", "endtime"))
        else:
            intervals = None
            
        assert attributes is not None or events is not None or intervals is not None, "At least one of attributes, events, or intervals must be provided"
        if attributes is not None:
            ids = attributes.get_ids()
        elif events is not None:
            ids = np.array(list(set(events.get_ids())))
        elif intervals is not None:
            ids = np.array(list(set(intervals.get_ids())))
            
        if sample:
            np.random.seed(1234)
            valid_ids = np.random.choice(ids, size=5000, replace=False)
        else:
            valid_ids = ids
        if attributes is not None:
            attributes = attributes.filter(attributes.get_ids().isin(valid_ids))
        if events is not None:
            events = events.filter(events.get_ids().isin(valid_ids))
        if intervals is not None:
            intervals = intervals.filter(intervals.get_ids().isin(valid_ids))
        
        if not os.path.exists(f"{self.data_dir}/train_test_split.pkl"):
            train_ids, val_ids = train_test_split(valid_ids, train_size=0.333)
            val_ids, test_ids = train_test_split(val_ids, test_size=0.5)
            with open(f"{self.data_dir}/train_test_split.pkl", "wb") as file:
                pickle.dump((train_ids, val_ids, test_ids), file)
        else:
            with open(f"{self.data_dir}/train_test_split.pkl", "rb") as file:
                train_ids, val_ids, test_ids = pickle.load(file)

        if val_only:
            if attributes is not None:
                attributes = attributes.filter(attributes.get_ids().isin(val_ids))
            if events is not None:
                events = events.filter(events.get_ids().isin(val_ids))
            if intervals is not None:
                intervals = intervals.filter(intervals.get_ids().isin(val_ids))

        if "macros" in data_config:
            macro_path = (os.path.join(self.base_path, data_config["macros"]["path"])
                          if "path" in data_config["macros"] 
                          else f"{self.data_dir}/macros.json")
            with open(macro_path, "r") as file:
                macros = json.load(file)
        else:
            macros = {}
            
        dataset = TrajectoryDataset(attributes, 
                                    events, 
                                    intervals, 
                                    cache_dir=cache_dir or os.path.join(self.cache_dir(), "variable_cache"), 
                                    eventtype_macros=macros)
        
        return dataset, (train_ids, val_ids, test_ids)

    def get_default_model_spec(self):
        model_info = self.config.get("models", {})
        default_info = model_info.get("default", {})
        return {
            "variables": default_info.get("variables", {"Untitled": {
                "category": "Inputs",
                "query": ""
            }}),
            "timestep_definition": default_info.get("timestep_definition", ""),
            "cohort": default_info.get("cohort", ""),
            "outcome": default_info.get("outcome", ""),
        }
        
    def cache_dir(self):
        cache_dir = os.path.join(self.base_path, "_cache")
        if not os.path.exists(cache_dir): os.mkdir(cache_dir)
        return cache_dir
        
    def model_spec_path(self, model_name):
        return os.path.join(self.model_dir, f"spec_{model_name}.json")
    
    def model_metrics_path(self, model_name):
        return os.path.join(self.model_dir, f"metrics_{model_name}.json")
    
    def model_preds_path(self, model_name):
        return os.path.join(self.model_dir, f"preds_{model_name}.pkl")
    
    def model_weights_path(self, model_name):
        return os.path.join(self.model_dir, f"model_{model_name}.json")
    
    def make_trainer(self):
        return ModelTrainer(self.config,
                            self.data_dir,
                            self.model_dir,
                            self.slices_dir)

    def make_slice_discovery_helper(self):
        return SliceDiscoveryHelper(self, 
                                    self.model_dir, 
                                    self.slices_dir, 
                                    **self.config["slices"].get("sampler", {}))
        
    def make_slice_evaluation_helper(self):
        return SliceEvaluationHelper(self,
                                     self.model_dir,
                                     self.slices_dir,
                                     **self.config["slices"].get("sampler", {}))