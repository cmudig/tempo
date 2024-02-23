import pandas as pd
import os
import re
import json
import tqdm
import torch
import pickle
import datetime
import time
from query_language.data_types import *
from model_training import make_query
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, r2_score, f1_score
import slice_finding as sf

MODEL_CACHE_UPDATE_TIME = 300 # 5 minutes

def make_slicing_variables(manager, dataset, variable_definitions, timestep_definition):
    """Creates the slicing variables dataframe."""
    query = make_query(variable_definitions, timestep_definition)
    print("Generated query")
    # Save the discretization mapping to a cache file so we can recover it later
    discretization_results = {}
    def get_unique_values(var_exp):
        value_indexes, col_spec = sf.discretization.discretize_column(var_exp.name, 
                                                                      var_exp.get_values(), 
                                                                      {"method": "unique"})
        if var_exp.name:
            discretization_results[var_exp.name] = col_spec
        return var_exp.with_values(pd.Series(value_indexes, name=var_exp.name))
        
    variable_df = dataset.query(query, variable_transform=("unique_values", get_unique_values))
    
    # Save the value name dictionary to a cache file
    discretization_path = os.path.join(manager.cache_dir(), "slice_discretizations", f"{timestep_definition}.pkl")
    if not os.path.exists(os.path.dirname(discretization_path)):
        os.mkdir(os.path.dirname(discretization_path))
        
    discrete_df = None
    if not discretization_results:
        try:
            with open(discretization_path, "rb") as file:
                discretization_results = pickle.load(file)
        except:
            print("Re-discretizing data because value names were not found")
            discrete_df = sf.discretization.discretize_data(variable_df.values, {
                col: { "method": "keep" }
                for col in variable_df.values.columns
            })
    else:
        with open(discretization_path, "wb") as file:
            pickle.dump(discretization_results, file)
    
    if discrete_df is None:
        dataframe = variable_df.values
        value_names = {i: discretization_results[col] 
                       for i, col in enumerate(dataframe.columns)
                       if col in discretization_results}
        # Find any keys that might not have been cached and re-discretize them
        missing_keys = [c for c in dataframe.columns if c not in discretization_results]
        if missing_keys:
            processed_missing_df = sf.discretization.discretize_data(dataframe[missing_keys], {
                col: { "method": "unique"} for col in missing_keys
            })
            reverse_index = {c: i for i, c in enumerate(dataframe.columns)}
            value_names.update({reverse_index[missing_keys[i]]: v for i, v in  processed_missing_df.value_names.items()})
            dataframe = dataframe.assign(**{c: processed_missing_df.df[:,i] for i, c in enumerate(missing_keys)})
        discrete_df = sf.discretization.DiscretizedData(dataframe.values.astype(np.uint8), value_names)
        
    print("Completed discretization")
    return discrete_df, variable_df.index.get_ids()

HOLDOUT_FRACTION = 0.5

def get_slicing_split(slices_dir, dataset, timestep_definition):
    split_path = os.path.join(slices_dir, "slicing_split.pkl")
    if os.path.exists(split_path):
        with open(split_path, "rb") as file:
            splits = pickle.load(file)
    else:
        splits = {}
        
    if timestep_definition not in splits:
        dataset_size = dataset.df.shape[0] if isinstance(dataset, sf.slices.DiscretizedData) else len(dataset)
        discovery_mask = (np.random.uniform(size=dataset_size) >= HOLDOUT_FRACTION)
        eval_mask = ~discovery_mask
        splits[timestep_definition] = (discovery_mask, eval_mask)
        with open(os.path.join(slices_dir, "slicing_split.pkl"), "wb") as file:
            pickle.dump(splits, file)
    
    return splits[timestep_definition]

def parse_controls(discrete_df, controls, source_mask=None):
    new_score_fns = {}
    initial_slice = None
    raw_inputs = discrete_df.df if hasattr(discrete_df, 'df') else discrete_df
    new_source_mask = source_mask.copy() if source_mask is not None else np.ones(len(raw_inputs), dtype=bool)
    exclusion_criteria = None
    
    if (contains_slice := controls.get("contains_slice", {})) != {}:
        contains_slice = discrete_df.encode_slice(contains_slice)
        if contains_slice.feature != sf.slices.SliceFeatureBase():
            ref_mask = contains_slice.make_mask(raw_inputs).cpu().numpy()
            new_score_fns["Contains Slice"] = sf.scores.SliceSimilarityScore(ref_mask, metric='superslice')
            exclusion_criteria = sf.filters.ExcludeIfAny([
                sf.filters.ExcludeFeatureValueSet([f.feature_name], f.allowed_values)
                for f in contains_slice.univariate_features()
            ])
            new_source_mask &= ref_mask

    if (contained_in_slice := controls.get("contained_in_slice", {})) != {}:
        contained_in_slice = discrete_df.encode_slice(contained_in_slice)
        if contained_in_slice.feature != sf.slices.SliceFeatureBase():
            ref_mask = contained_in_slice.make_mask(raw_inputs).cpu().numpy()
            new_score_fns["Contained in Slice"] = sf.scores.SliceSimilarityScore(ref_mask, metric='subslice')
            exclusion_criteria = sf.filters.ExcludeIfAny([
                sf.filters.ExcludeFeatureValueSet([f.feature_name], f.allowed_values)
                for f in contained_in_slice.univariate_features()
            ])
            new_source_mask &= ref_mask

    if (similar_to_slice := controls.get("similar_to_slice", {})) != {}:
        similar_to_slice = discrete_df.encode_slice(similar_to_slice)
        if similar_to_slice.feature != sf.slices.SliceFeatureBase():
            ref_mask = similar_to_slice.make_mask(raw_inputs).cpu().numpy()
            new_score_fns["Similar to Slice"] = sf.scores.SliceSimilarityScore(ref_mask, metric='jaccard')
            exclusion_criteria = sf.filters.ExcludeIfAny([
                sf.filters.ExcludeFeatureValueSet([f.feature_name], f.allowed_values)
                for f in similar_to_slice.univariate_features()
            ])
            new_source_mask &= ref_mask

    if (subslice_of_slice := controls.get("subslice_of_slice", {})) != {}:
        subslice_of_slice = discrete_df.encode_slice(subslice_of_slice)
        if subslice_of_slice.feature != sf.slices.SliceFeatureBase():
            initial_slice = subslice_of_slice
        
    return new_score_fns, initial_slice, new_source_mask, exclusion_criteria

def default_control_weights(controls):
    weights = {}
    if controls.get("contains_slice", {}) != {}:
        weights["Contains Slice"] = 1.0

    if controls.get("contained_in_slice", {}) != {}:
        weights["Contained in Slice"] = 1.0

    if controls.get("similar_to_slice", {}) != {}:
        weights["Similar to Slice"] = 1.0

    return weights        
    
def filter_single_values(valid_df, outcomes):
    # Exclude any features that have only one value
    single_value_filters = {}
    for col_idx, (col, value_pairs) in valid_df.value_names.items():
        unique_vals = np.unique(valid_df.df[:,col_idx][~pd.isna(outcomes)])
        if len(unique_vals) <= 1:
            single_value_filters.setdefault(unique_vals[0], []).append(col_idx)
            
    print("Single value filters:", single_value_filters)
    if single_value_filters:
        return sf.filters.ExcludeIfAny([sf.filters.ExcludeFeatureValueSet(cols, [v]) for v, cols in single_value_filters.items()])
    
def find_slices(discrete_df, score_fns, controls=None, progress_fn=None, n_samples=100, seen_slices=None, **kwargs):
    """
    Returns a tuple containing (metrics dict, slices list).
    """
    
    finder = sf.sampling.SamplingSliceFinder(
        discrete_df, 
        score_fns,
        holdout_fraction=0.0,
        **kwargs
    )
    
    if controls is not None:
        (new_score_fns, 
         initial_slice, 
         new_source_mask, 
         exclusion_criteria) = parse_controls(discrete_df, controls, source_mask=finder.source_mask)
        if initial_slice is None: initial_slice = finder.initial_slice
        
        new_filter = finder.group_filter
        if exclusion_criteria is not None:
            if new_filter is not None:
                new_filter = sf.filters.ExcludeIfAny([new_filter, exclusion_criteria])
            else:
                new_filter = exclusion_criteria
        finder = finder.copy_spec(
            score_fns={**finder.score_fns, **new_score_fns},
            source_mask=finder.source_mask & new_source_mask if finder.source_mask is not None else new_source_mask,
            group_filter=new_filter,
            initial_slice=initial_slice,
        )
    
    if seen_slices is not None:
        finder.seen_slices = {**seen_slices}
        finder.all_scores += list(seen_slices.keys())
    
    finder.progress_fn = progress_fn
    results, _ = finder.sample(n_samples)
    
    return results.results

class SliceHelper:
    def __init__(self, manager, model_dir, results_dir):
        self.manager = manager
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.discrete_dfs = {}
        
        self.last_cache_time = None
        self.model_spec_cache = {}
        self.model_preds_cache = {}
        self.model_threshold_cache = {} # just for binary classification models
        
    def get_status(self):
        status_path = os.path.join(self.results_dir, "discovery_status.json")
        try:
            if os.path.exists(status_path):
                with open(status_path, "r") as file:
                    return json.load(file)
        except:
            # Some race condition is happening here?
            print("EXCEPTION GETTING STATUS")
            pass
        return {"searching": False, "status": {"state": "none", "message": "Not currently finding slices"}, "n_results": 0, "n_runs": 0, "models": []}
        
    def get_raw_slicing_dataset(self):
        dataset, _ = self.manager.load_data(cache_dir=os.path.join(self.manager.cache_dir(), "slicing_variables"), val_only=True)
        return dataset
        
    def get_slicing_data(self, slice_spec_name, timestep_def, evaluation=False):
        if (slice_spec_name, timestep_def, evaluation) not in self.discrete_dfs:
            dataset = self.get_raw_slicing_dataset()
            
            if slice_spec_name == "default":
                slicing_metadata = self.manager.get_default_slice_spec()
            else:
                with open(os.path.join(self.results_dir, "specifications", slice_spec_name + ".json"), "r") as file:
                    slicing_metadata = json.load(file)
                
            discrete_df, ids = make_slicing_variables(self.manager, dataset, slicing_metadata["variables"], timestep_def) 
            row_mask = get_slicing_split(self.results_dir, dataset, timestep_def)[1 if evaluation else 0]
            valid_df = discrete_df.filter(row_mask)
            ids = ids.values[row_mask]
            if "slice_filter" in slicing_metadata:
                slice_filter = valid_df.encode_filter(sf.filters.SliceFilterBase.from_dict(slicing_metadata["slice_filter"]))
                print(slice_filter)
            else:
                slice_filter = sf.filters.SliceFilterBase()
            self.discrete_dfs[(slice_spec_name, timestep_def, evaluation)] = (valid_df, row_mask, ids, slice_filter)
        return self.discrete_dfs[(slice_spec_name, timestep_def, evaluation)]
        
    def invalidate_slice_spec(self, slice_spec_name):
        """
        Removes all data artifacts and found slices belonging to the given slice spec.
        """
        self.discrete_dfs = {k: v for k, v in self.discrete_dfs.items() if k[0] != slice_spec_name}

    def clear_model_caches(self):
        self.model_spec_cache = {}
        self.model_preds_cache = {}
        self.model_threshold_cache = {}
        self.last_cache_time = time.time()
        
    def get_model_spec(self, model_name):
        if self.last_cache_time is None or time.time() - self.last_cache_time >= MODEL_CACHE_UPDATE_TIME:
            self.clear_model_caches()
            
        if model_name not in self.model_spec_cache:
            with open(self.manager.model_spec_path(model_name), "r") as file:
                self.model_spec_cache[model_name] = json.load(file)
        return self.model_spec_cache[model_name]
        
    def get_model_preds(self, model_name):
        """Returns a tuple of (true values, predicted values)."""
        if self.last_cache_time is None or time.time() - self.last_cache_time >= MODEL_CACHE_UPDATE_TIME:
            self.clear_model_caches()
            
        if model_name not in self.model_preds_cache:
            with open(self.manager.model_preds_path(model_name), "rb") as file:
                self.model_preds_cache[model_name] = tuple((x.astype(np.float64) for x in pickle.load(file)))
        return self.model_preds_cache[model_name]
    
    def get_model_opt_threshold(self, model_name):
        if self.last_cache_time is None or time.time() - self.last_cache_time >= MODEL_CACHE_UPDATE_TIME:
            self.clear_model_caches()
            
        if model_name not in self.model_threshold_cache:
            with open(self.manager.model_metrics_path(model_name), "r") as file:
                self.model_threshold_cache[model_name] = json.load(file)["threshold"]
        return self.model_threshold_cache[model_name]        
        
    def get_valid_model_mask(self, model_name):
        return ~np.isnan(self.get_model_preds(model_name)[0])
    
    def get_model_labels(self, model_name):
        return self.get_model_preds(model_name)[0]
            
    def get_score_functions(self, discrete_df, valid_mask, timestep_def, include_model_names=None, exclude_model_names=None, controls=None):
        score_fns = {
            "size": sf.scores.SliceSizeScore(0.2, 0.05),
            "complexity": sf.scores.NumFeaturesScore()
        }
        model_names = []
        for path in os.listdir(self.model_dir):
            if not path.startswith("preds"): continue
            model_name = re.search(r"^preds_(.*).pkl$", path).group(1)
            spec = self.get_model_spec(model_name)
            if spec["timestep_definition"] != timestep_def:
                continue
            if include_model_names is not None and model_name not in include_model_names: continue
            if exclude_model_names is not None and model_name in exclude_model_names: continue
            
            model_names.append(model_name)
            
            outcomes, preds = self.get_model_preds(model_name)
            valid_outcomes = outcomes[valid_mask].astype(np.float64)
            
            if spec["model_type"] == "binary_classification":
                threshold = self.get_model_opt_threshold(model_name)
                preds = np.where(np.isnan(preds), np.nan, preds >= threshold)
                
                valid_preds = preds[valid_mask]
                valid_true = np.where(np.isnan(valid_outcomes), np.nan, valid_outcomes > 0)
                
                valid_acc = np.where(np.isnan(valid_outcomes), np.nan, (valid_outcomes > 0) == valid_preds)
                score_fns.update({
                    f"{model_name}_true": sf.scores.OutcomeRateScore(valid_true),
                    f"{model_name}_true_low": sf.scores.OutcomeRateScore(valid_true, inverse=True),
                    f"{model_name}_pred": sf.scores.OutcomeRateScore(valid_preds),
                    f"{model_name}_pred_low": sf.scores.OutcomeRateScore(valid_preds, inverse=True),
                    f"{model_name}_err": sf.scores.OutcomeRateScore(1 - valid_acc),
                    f"{model_name}_err_low": sf.scores.OutcomeRateScore(1 - valid_acc, inverse=True),
                })

            elif spec["model_type"] == "multiclass_classification":
                valid_preds = np.where(np.isnan(preds).max(axis=1), np.nan, np.argmax(preds, axis=1))[valid_mask]
                
                valid_acc = np.where(np.isnan(valid_outcomes), np.nan, valid_outcomes == valid_preds)
                score_fns.update({
                    f"{model_name}_err": sf.scores.OutcomeRateScore(1 - valid_acc),
                    f"{model_name}_err_low": sf.scores.OutcomeRateScore(1 - valid_acc),
                })
                if preds.shape[1] <= 10:
                    # Add score functions for every single category
                    for i, output_value in enumerate(spec["output_values"]):
                        valid_true_value = np.where(np.isnan(valid_outcomes), np.nan, valid_outcomes == i)
                        valid_pred_value = np.where(np.isnan(valid_preds), np.nan, valid_preds == i)
                        score_fns.update({
                            f"{model_name}_=_{output_value}_true": sf.scores.OutcomeRateScore(valid_true_value),
                            f"{model_name}_=_{output_value}_true_low": sf.scores.OutcomeRateScore(valid_true_value, inverse=True),
                            # f"{model_name}_{output_value}_true_xf": sf.scores.InteractionEffectScore(valid_true_value),
                            f"{model_name}_=_{output_value}_pred": sf.scores.OutcomeRateScore(valid_pred_value),
                            f"{model_name}_=_{output_value}_pred_low": sf.scores.OutcomeRateScore(valid_pred_value, inverse=True),
                            # f"{model_name}_{output_value}_pred_xf": sf.scores.InteractionEffectScore(valid_pred_value),
                        })
            elif spec["model_type"] == "regression":
                valid_preds = preds[valid_mask]
                
                valid_diff = np.where(np.isnan(valid_outcomes), np.nan, np.abs(valid_outcomes - valid_preds))
                score_fns.update({
                    f"{model_name}_diff_mean": sf.scores.MeanDifferenceScore(valid_diff),
                    f"{model_name}_true_above": sf.scores.OutcomeRateScore(valid_outcomes > np.nanmean(valid_outcomes)),
                    f"{model_name}_true_below": sf.scores.OutcomeRateScore(valid_outcomes < np.nanmean(valid_outcomes)),
                    f"{model_name}_pred_above": sf.scores.OutcomeRateScore(valid_preds > np.nanmean(valid_preds)),
                    f"{model_name}_pred_below": sf.scores.OutcomeRateScore(valid_preds < np.nanmean(valid_preds))
                })
                            
        if controls is not None:
            new_score_fns, _, _, _ = parse_controls(discrete_df, controls)
            score_fns.update(new_score_fns)
        return score_fns, model_names

    def get_model_timestep_def(self, model_name):
        return self.get_model_spec(model_name)['timestep_definition']
    
    def controls_to_result_key(self, controls):
        """Returns a key into the slice results file representing the results for
        this set of controls."""
        def make_slice(obj):
            if obj is None or len(obj) == 0: return None
            return sf.slices.SliceFeatureBase.from_dict(obj)
        return (controls["slice_spec_name"],
                make_slice(controls.get("contains_slice", None)),
                make_slice(controls.get("contained_in_slice", None)),
                make_slice(controls.get("similar_to_slice", None)),
                make_slice(controls.get("subslice_of_slice", None)))
        
    def result_key_to_controls(self, result_key):
        """Returns a key into the slice results file representing the results for
        this set of controls."""
        return {"slice_spec_name": result_key[0],
                "contains_slice": result_key[1].to_dict() if result_key[1] is not None else None,
                "contained_in_slice": result_key[2].to_dict() if result_key[2] is not None else None,
                "similar_to_slice": result_key[3].to_dict() if result_key[3] is not None else None,
                "subslice_of_slice": result_key[4].to_dict() if result_key[4] is not None else None}
    
    def load_timestep_slice_results(self, timestep_def):
        results_path = os.path.join(self.results_dir, f"slice_results_{timestep_def}.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as file:
                results = json.load(file)["results"]
                self.slice_scores[timestep_def] = {
                    self.controls_to_result_key(item["controls"]): 
                        {s: s.score_values for s in (sf.slices.Slice.from_dict(r) for r in item["slices"])}
                    for item in results
                }
               
class SliceDiscoveryHelper(SliceHelper):
    def __init__(self, manager, model_dir, results_dir, samples_per_model=50, min_items_fraction=0.01, **slice_finding_kwargs):
        super().__init__(manager, model_dir, results_dir)
        self.samples_per_model = samples_per_model
        self.min_items_fraction = min_items_fraction
        self.slice_finding_kwargs = slice_finding_kwargs
        self.slice_scores = {}
        
    def write_status(self, searching, timestep_defs_updated=None, search_status=None, new_results=0, new_runs=0, models_to_add=None, models_to_remove=None, error_model=None, error_message=None):
        status = self.get_status()
        status["searching"] = searching
        if search_status is not None:
            status["status"] = search_status
            if "model_name" in search_status and search_status["model_name"] in status.get("errors", {}):
                del status["errors"][search_status["model_name"]]
        elif "status" in status:
            status["status"] = {"state": "none", "message": "Not currently finding slices"}
        status["n_results"] = status["n_results"] + new_results
        status["n_runs"] = status["n_runs"] + new_runs
        status["models"] = [m for m in [*status["models"], *(models_to_add if models_to_add is not None else [])]
                            if models_to_remove is None or m not in models_to_remove]
        for model in status["models"]:
            if model in status.get("errors", {}):
                del status["errors"][error_model]
        if error_model is not None and error_message is not None:
            status.setdefault("errors", {})
            status["errors"][error_model] = error_message
        if timestep_defs_updated is not None:
            status["last_updated"] = {**status.get("last_updated", {}), 
                                      **{td: str(datetime.datetime.now()) for td in timestep_defs_updated}}
        status_path = os.path.join(self.results_dir, "discovery_status.json")
        with open(status_path, "w") as file:
            json.dump(status, file)
             
    def save_timestep_slice_results(self, timestep_def):
        results_path = os.path.join(self.results_dir, f"slice_results_{timestep_def}.json")
        with open(results_path, "w") as file:
            json.dump(sf.utils.convert_to_native_types({"results": [{
                "controls": self.result_key_to_controls(result_key),
                "slices": [r.to_dict() for r in results]
            } for result_key, results in self.slice_scores[timestep_def].items()]}), file)
        
    def model_has_slices(self, model_name):
        """
        Determines whether there are slices for a given model's timestep 
        definition.
        """
        timestep_def = self.get_model_timestep_def(model_name)
        results_path = os.path.join(self.results_dir, f"slice_results_{timestep_def}.json")
        return os.path.exists(results_path)
        
    def rescore_model(self, model_name):
        """Recalculates the discovery set scores for the given model."""
        timestep_def = self.get_model_timestep_def(model_name)
        self.load_timestep_slice_results(timestep_def)   
        if not self.slice_scores.get(timestep_def, []):
            print("No slices to rescore")
            return
        
        for control_set, control_results in self.slice_scores[timestep_def].items():
            valid_df, discovery_mask, _, _ = self.get_slicing_data(self.result_key_to_controls(control_set)["slice_spec_name"], 
                                                                   timestep_def)
            
            other_score_fns, scored_model_names = self.get_score_functions(valid_df, discovery_mask, timestep_def, include_model_names=[model_name])
            results = list(control_results.keys())
            min_items = self.min_items_fraction * len(valid_df.df)
            print(len(results), "results to rescore")
            rescored_results = sf.slices.score_slices_batch(results, 
                                                            valid_df.df, 
                                                            other_score_fns, 
                                                            self.slice_finding_kwargs.get("max_features", 3), 
                                                            min_items=min_items, 
                                                            device='cpu')
            print(len(rescored_results), "slices scored")
            
            for r in results:
                r = r.rescore({**r.score_values, **(rescored_results[r].score_values if rescored_results.get(r, None) is not None else {})})
                if r in control_results:
                    del control_results[r]
                control_results[r] = r.score_values
        
        print("Writing to file")
        self.save_timestep_slice_results(timestep_def)
        self.write_status(False, timestep_defs_updated=[timestep_def])
        print("Done")
        
    def invalidate_slice_spec(self, slice_spec_name):
        super().invalidate_slice_spec(slice_spec_name)
        # Remove all found slices that have this slice spec name
        for path in os.listdir(self.results_dir):
            if (match := re.match(r"^slice_results_(.*).json$"), path) is not None:
                timestep_def = match.group(1)
                self.load_timestep_slice_results(timestep_def)
                self.slice_scores[timestep_def] = {
                    k: v for k, v in self.slice_scores[timestep_def].items()
                    if self.result_key_to_controls(k)['slice_spec_name'] != slice_spec_name
                }
        
    def find_slices(self, model_name, controls, additional_filter=None):
        try:
            self.write_status(True, search_status={"state": "loading", "message": "Loading data", "model_name": model_name})
            
            spec = self.get_model_spec(model_name)
            timestep_def = spec["timestep_definition"]
            self.load_timestep_slice_results(timestep_def)
            self.write_status(True, search_status={"state": "loading", "message": "Loading variables", "model_name": model_name})
            valid_df, discovery_mask, _, slice_filter = self.get_slicing_data(controls["slice_spec_name"], timestep_def)
                 
            self.write_status(True, search_status={"state": "loading", "message": f"Finding slices for {model_name}", "model_name": model_name})
            
            outcomes = self.get_model_labels(model_name)
                
            # don't add control-related score functions here as they will be added
            # in find_slices
            discovery_score_fns, _ = self.get_score_functions(valid_df, discovery_mask, timestep_def, include_model_names=[model_name])
            
            result_key = self.controls_to_result_key(controls)
            print('result key:', result_key)
            
            # Save time by only finding slices on the rows where the outcome exists
            outcome_exists = ~np.isnan(outcomes[discovery_mask])
            discovery_df = valid_df.filter(outcome_exists)
            discovery_score_fns = {k: fn.subslice(outcome_exists) for k, fn in discovery_score_fns.items()}
            discovery_outcomes = outcomes[discovery_mask][outcome_exists]
            
            last_progress = 0
            
            def update_sampler_progress(progress, total):
                nonlocal last_progress
                self.write_status(True, 
                                  search_status={"state": "loading", "message": f"Finding slices for {model_name} ({progress} / {total})", "progress": progress / total, "model_name": model_name})
                last_progress = progress
            
            print(len(discovery_outcomes), "outcomes")
            min_items = self.min_items_fraction * len(discovery_df.df)
            
            # Add filters
            if additional_filter is not None: 
                if callable(additional_filter):
                    slice_filter = sf.filters.ExcludeIfAny([slice_filter, additional_filter(valid_df, outcomes[discovery_mask])])
                else:
                    slice_filter = sf.filters.ExcludeIfAny([slice_filter, additional_filter])
            single_value_filter = filter_single_values(valid_df, outcomes[discovery_mask])
            if single_value_filter is not None:
                slice_filter = sf.filters.ExcludeIfAny([slice_filter, single_value_filter])
                
            self.slice_scores.setdefault(timestep_def, {})
            command_results = self.slice_scores[timestep_def].setdefault(result_key, {})
            print(discovery_df.df.shape, min_items, self.slice_finding_kwargs)

            if spec["model_type"].endswith("classification"):
                # Randomly select a subset of rows such that the number of elements
                # with each outcome value is roughly constant
                unique_values, unique_counts = np.unique(discovery_outcomes, return_counts=True)
                sample_count = max(np.min(unique_counts), int(len(discovery_outcomes) * 0.01))
                source_mask = np.zeros(len(discovery_outcomes), dtype=bool)
                for val, count in zip(unique_values, unique_counts):
                    matching_idxs = np.arange(len(discovery_outcomes), dtype=np.uint32)[discovery_outcomes == val]
                    source_mask[np.random.choice(matching_idxs, size=min(count, sample_count), replace=False)] = True
                print(np.unique(discovery_outcomes[source_mask], return_counts=True))
            else:
                source_mask = None
            
            results = find_slices(discovery_df, 
                                    discovery_score_fns,
                                    controls=controls,
                                    progress_fn=update_sampler_progress, 
                                    seen_slices=command_results,
                                    n_samples=self.samples_per_model, 
                                    min_items=min_items,
                                    n_workers=None,
                                    source_mask=source_mask,
                                    group_filter=slice_filter,
                                    **self.slice_finding_kwargs)

            # Add scores for all the other models
            other_score_fns, scored_model_names = self.get_score_functions(valid_df, discovery_mask, timestep_def)
            new_results = [r for r in results if r not in command_results]
            rescored_results = sf.slices.score_slices_batch(new_results, 
                                                            valid_df.df, 
                                                            other_score_fns, 
                                                            self.slice_finding_kwargs.get("max_features", 3), 
                                                            min_items=min_items, 
                                                            device='cpu')
            
            for r in new_results:
                r = r.rescore({**r.score_values, **(rescored_results[r].score_values if rescored_results.get(r, None) is not None else {})})
                if r in command_results:
                    del command_results[r]
                command_results[r] = r.score_values
            
            self.save_timestep_slice_results(timestep_def)
            self.write_status(False, timestep_defs_updated=[timestep_def], new_runs=self.samples_per_model, new_results=len(results), models_to_add=[model_name])
        except KeyboardInterrupt:
            self.write_status(False, search_status=None)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.write_status(False, error_model=model_name, error_message=str(e))
            
class SliceEvaluationHelper(SliceHelper):
    def __init__(self, manager, model_dir, results_dir, min_items_fraction=0.01, **slice_finding_kwargs):
        super().__init__(manager, model_dir, results_dir)
        self.min_items_fraction = min_items_fraction
        self.slice_finding_kwargs = slice_finding_kwargs
        self.slice_finding_status = None
        self.slice_scores = {}
        self.eval_score_caches = {}
        self.metrics = {}
        self.eval_ids = {} 
        
    def get_default_evaluation_weights(self, model_names, controls=None):
        if controls is not None:
            control_weights = default_control_weights(controls)
        else:
            control_weights = {}
        eval_weights = {**control_weights, "size": 0.5, "complexity": 0.5}
        for model_name in model_names:
            spec = self.get_model_spec(model_name)
            if spec["model_type"] == "binary_classification":
                eval_weights.update({n: 1.0 for model_name in model_names
                   for n in [f"{model_name}_true",
                             f"{model_name}_pred"]
                })
                eval_weights.update({n: 0.0 for model_name in model_names
                   for n in [f"{model_name}_true_low",
                             f"{model_name}_pred_low",
                             f"{model_name}_err",
                             f"{model_name}_err_low",]
                })
            elif spec["model_type"] == "multiclass_classification":
                eval_weights[f"{model_name}_err"] = 0.0
                eval_weights[f"{model_name}_err_low"] = 1.0
                for val in spec["output_values"]:
                    eval_weights[f"{model_name}_=_{val}_true"] = 0.0
                    eval_weights[f"{model_name}_=_{val}_true_low"] = 0.0
                    eval_weights[f"{model_name}_=_{val}_pred"] = 0.0
                    eval_weights[f"{model_name}_=_{val}_pred_low"] = 0.0
            else:
                eval_weights[f"{model_name}_true_above"] = 1.0
                eval_weights[f"{model_name}_pred_above"] = 1.0
                eval_weights[f"{model_name}_true_below"] = 0.0
                eval_weights[f"{model_name}_pred_below"] = 0.0
                eval_weights[f"{model_name}_diff_mean"] = 0.0
        return eval_weights
        
    def weights_for_evaluation(self, display_weights):
        """Converts the score weights to the names of the actual score functions."""
        new_weights = {}
        for wname, w in display_weights.items():
            if (match := re.match(r'^Negative Label (.*)$', wname)) is not None:
                model_name = match.group(1)
                new_weights[f"{model_name}_true_low"] = w
                # new_weights[f"{model_name}_true_xf"] = w
            elif (match := re.match(r'^Positive Label (.*)$', wname)) is not None:
                model_name = match.group(1)
                new_weights[f"{model_name}_true"] = w
                # new_weights[f"{model_name}_true_xf"] = w
            elif (match := re.match(r'^Negative Prediction (.*)$', wname)) is not None:
                model_name = match.group(1)
                new_weights[f"{model_name}_pred_low"] = w
                # new_weights[f"{model_name}_pred_xf"] = w
            elif (match := re.match(r'^Positive Prediction (.*)$', wname)) is not None:
                model_name = match.group(1)
                new_weights[f"{model_name}_pred"] = w
                # new_weights[f"{model_name}_pred_xf"] = w
            elif (match := re.match(r"^Label '(.*)' != '(.*)'$", wname)) is not None:
                model_name = match.group(1)
                value = match.group(2)
                new_weights[f"{model_name}_=_{value}_true_low"] = w
            elif (match := re.match(r"^Label '(.*)' = '(.*)'$", wname)) is not None:
                model_name = match.group(1)
                value = match.group(2)
                new_weights[f"{model_name}_=_{value}_true"] = w
            elif (match := re.match(r"^Prediction '(.*)' != '(.*)'$", wname)) is not None:
                model_name = match.group(1)
                value = match.group(2)
                new_weights[f"{model_name}_=_{value}_pred_low"] = w
            elif (match := re.match(r"^Prediction '(.*)' = '(.*)'$", wname)) is not None:
                model_name = match.group(1)
                value = match.group(2)
                new_weights[f"{model_name}_=_{value}_pred"] = w
            elif (match := re.match(r'^Difference (.*)$', wname)) is not None:
                model_name = match.group(1)
                new_weights[f"{model_name}_diff_mean"] = w
            elif (match := re.match(r'^High Label (.*)$', wname)) is not None:
                model_name = match.group(1)
                new_weights[f"{model_name}_true_above"] = w
            elif (match := re.match(r'^High Prediction (.*)$', wname)) is not None:
                model_name = match.group(1)
                new_weights[f"{model_name}_pred_above"] = w
            elif (match := re.match(r'^Low Label (.*)$', wname)) is not None:
                model_name = match.group(1)
                new_weights[f"{model_name}_true_below"] = w
            elif (match := re.match(r'^Low Prediction (.*)$', wname)) is not None:
                model_name = match.group(1)
                new_weights[f"{model_name}_pred_below"] = w
            elif (match := re.match(r'^Accuracy (.*)$', wname)) is not None:
                model_name = match.group(1)
                new_weights[f"{model_name}_err_low"] = w
                # new_weights[f"{model_name}_acc_xf"] = w
            elif (match := re.match(r'^Error (.*)$', wname)) is not None:
                model_name = match.group(1)
                new_weights[f"{model_name}_err"] = w
                # new_weights[f"{model_name}_err_xf"] = w
            elif wname == "complexity": new_weights["Simple Rule"] = w
            elif wname == "size": new_weights["Large Slice"] = w
            else:
                new_weights[wname] = w
        return new_weights
    
    def weights_for_display(self, evaluation_weights):
        """Converts the score weights to user-readable weight values."""
        new_weights = {}
        for wname, w in evaluation_weights.items():
            display_key = wname
            if (match := re.match(r'^(.*)_=_(.*)_true(_xf)?$', wname)) is not None:
                display_key = f"Label '{match.group(1)}' = '{match.group(2)}'"
            elif (match := re.match(r'^(.*)_=_(.*)_true_low?$', wname)) is not None:
                display_key = f"Label '{match.group(1)}' != '{match.group(2)}'"
            elif (match := re.match(r'^(.*)_=_(.*)_pred_low?$', wname)) is not None:
                display_key = f"Prediction '{match.group(1)}' != '{match.group(2)}'"
            elif (match := re.match(r'^(.*)_=_(.*)_pred(_xf)?$', wname)) is not None:
                display_key = f"Predicted '{match.group(1)}' = '{match.group(2)}'"
            elif (match := re.match(r'^(.*)_true(_xf)?$', wname)) is not None:
                display_key = f"Positive Label {match.group(1)}"
            elif (match := re.match(r'^(.*)_true_low$', wname)) is not None:
                display_key = f"Negative Label {match.group(1)}"
            elif (match := re.match(r'^(.*)_pred_low$', wname)) is not None:
                display_key = f"Negative Prediction {match.group(1)}"
            elif (match := re.match(r'^(.*)_pred(_xf)?$', wname)) is not None:
                display_key = f"Positive Prediction {match.group(1)}"
            elif (match := re.match(r'^(.*)_true_above$', wname)) is not None:
                display_key = f"High Label {match.group(1)}"
            elif (match := re.match(r'^(.*)_true_below$', wname)) is not None:
                display_key = f"Low Label {match.group(1)}"
            elif (match := re.match(r'^(.*)_pred_above$', wname)) is not None:
                display_key = f"High Prediction {match.group(1)}"
            elif (match := re.match(r'^(.*)_pred_below$', wname)) is not None:
                display_key = f"Low Prediction {match.group(1)}"
            elif (match := re.match(r'^(.*)_err_low$', wname)) is not None:
                display_key = f"Accuracy {match.group(1)}"
            elif (match := re.match(r'^(.*)_err(_xf)?$', wname)) is not None:
                display_key = f"Error {match.group(1)}"
            elif (match := re.match(r'^(.*)_diff_mean$', wname)) is not None:
                display_key = f"Difference {match.group(1)}"
            elif wname == "complexity": display_key = "Simple Rule"
            elif wname == "size": display_key = "Large Slice"
            new_weights[display_key] = new_weights.get(display_key, 0) + w
        return new_weights
        
    def invalidate_slice_spec(self, slice_spec_name):
        super().invalidate_slice_spec(slice_spec_name)
        # Remove all scored slices 
        for timestep_def in list(self.slice_scores.keys()):
            self.slice_scores[timestep_def] = {
                k: v for k, v in self.slice_scores[timestep_def].items()
                if self.result_key_to_controls(k)['slice_spec_name'] != slice_spec_name
            }
        for timestep_def in list(self.eval_score_caches.keys()):
            self.eval_score_caches[timestep_def] = {
                k: v for k, v in self.eval_score_caches[timestep_def].items()
                if self.result_key_to_controls(k)['slice_spec_name'] != slice_spec_name
            }
    
    def describe_slice(self, rank_list, metrics, ids, slice_obj, model_names, return_instance_info=False):
        """
        Generates a slice description of the given slice, adding count variables
        for each model outcome.
        """
        desc, mask = rank_list.generate_slice_description(slice_obj, metrics=metrics, return_slice_mask=True)
        old_desc_metrics = desc["metrics"]
        desc["metrics"] = {}
        
        instance_behaviors = {}
        for model_name in model_names:
            spec = self.get_model_spec(model_name)
            
            true_outcome = metrics[f"{model_name} True"]
            total_nonna = (~pd.isna(true_outcome)).sum()
            matching_nonna = (~pd.isna(true_outcome[mask])).sum()
            
            total_unique_ids = len(np.unique(ids[~pd.isna(true_outcome)]))
            
            pred_outcome = metrics[f"{model_name} Predicted"]
            full_slice_true = true_outcome[mask]
            full_slice_pred = pred_outcome[mask]
            slice_ids = ids[mask][~pd.isna(full_slice_true)]
            slice_pred = full_slice_pred[~pd.isna(full_slice_true)]
            slice_true = full_slice_true[~pd.isna(full_slice_true)]
            matching_unique_ids = len(np.unique(slice_ids))
            desc["metrics"][model_name] = {
                "Timesteps": {"type": "count", "count": matching_nonna, "share": matching_nonna / total_nonna},
                "Trajectories": {"type": "count", 
                                 "count": matching_unique_ids, 
                                 "share": matching_unique_ids / total_unique_ids},
            }
            if matching_nonna < self.min_items_fraction * total_nonna:
                continue
            desc["metrics"][model_name].update({
                "Labels": old_desc_metrics[f"{model_name} True"],
                "Predictions": old_desc_metrics[f"{model_name} Predicted"]
            })
            
            if spec["model_type"] == "binary_classification":
                opt_threshold = self.get_model_opt_threshold(model_name)
                    
                instance_behaviors[model_name] = (full_slice_true, np.where(pd.isna(full_slice_true), np.nan, full_slice_pred >= opt_threshold))
                if len(np.unique(slice_true)) <= 1:
                    auroc = 1
                else:
                    auroc = roc_auc_score(slice_true, slice_pred)
                if len(slice_pred) > 0:
                    conf = confusion_matrix(slice_true, (slice_pred >= opt_threshold), labels=[0, 1])
                    tn, fp, fn, tp = conf.ravel()
                    acc = ((slice_pred >= opt_threshold) == slice_true).mean()
                else:
                    tn, fp, fn, tp = 0, 1, 1, 0
                    acc = float('nan')
                             
                f1 = f1_score(slice_true, slice_pred >= opt_threshold)
                   
                desc["metrics"][model_name].update({
                    "Accuracy": {"type": "numeric", 
                                "value": acc},
                    "AUROC": {"type": "numeric", 
                                "value": auroc},
                    "Sensitivity": {"type": "numeric", 
                                "value": float(tp / (tp + fn))},
                    "Specificity": {"type": "numeric", 
                                "value": float(tn / (tn + fp))},
                    "Precision": {"type": "numeric", 
                                "value": float(tp / (tp + fp))},
                    "Micro F1": {"type": "numeric", 
                                "value": f1},
                    "Macro F1": {"type": "numeric", 
                                "value": f1},
                })
            elif spec["model_type"] == "multiclass_classification":
                desc["metrics"][model_name].update({
                    "Accuracy": {"type": "numeric", "value": float((slice_true == slice_pred).mean())},
                    "Micro F1": {"type": "numeric", "value": float(f1_score(slice_true, slice_pred, average="micro"))},
                    "Macro F1": {"type": "numeric", "value": float(f1_score(slice_true, slice_pred, average="macro"))},
                })
            else:
                desc["metrics"][model_name].update({
                    "R^2": {"type": "numeric", "value": float(r2_score(slice_true, slice_pred))},
                    "MSE": {"type": "numeric", "value": float(np.mean((slice_true - slice_pred) ** 2))}
                })
            
        # Remove scores that aren't related to these model names
        desc["scoreValues"] = {k: v for k, v in desc["scoreValues"].items() 
                               if k in ("size", "complexity") or any(k.startswith(model_name + "_") for model_name in model_names)}
        if return_instance_info: return desc, (mask, instance_behaviors)
        return desc
        
    def rescore_model(self, model_name, timestep_def=None):
        """Recalculates the evaluation slice scores for the given model."""
        print("Rescoring model")
        if model_name in self.metrics: 
            del self.metrics[model_name]
            
        # Clear slice score cache so that the slices will be rescored
        if timestep_def is None: timestep_def = self.get_model_timestep_def(model_name)
        self.eval_score_caches[timestep_def] = {}
        
    def get_eval_metrics(self, model_names, eval_mask):
        metrics = {}
        for model_name in model_names:
            if model_name not in self.metrics:
                spec = self.get_model_spec(model_name)
                
                model_metrics = {}
                outcomes, preds = self.get_model_preds(model_name)
            
                valid_outcomes = outcomes[eval_mask].astype(np.float64)
                valid_preds = preds[eval_mask]
                if len(valid_preds.shape) > 1: valid_preds = np.argmax(valid_preds, axis=1)

                if spec["model_type"] == "multiclass_classification":
                    output_vals = np.array(spec["output_values"])
                    true_metrics = np.empty(len(valid_outcomes), dtype=object)
                    true_metrics.fill(None)
                    true_metrics[~np.isnan(valid_outcomes)] = output_vals[valid_outcomes[~np.isnan(valid_outcomes)].astype(int)]
                    pred_metrics = np.empty(len(valid_outcomes), dtype=object)
                    pred_metrics.fill(None)
                    pred_metrics[~np.isnan(valid_preds)] = output_vals[valid_preds[~np.isnan(valid_preds)].astype(int)]
                    model_metrics[f"{model_name} True"] = true_metrics
                    model_metrics[f"{model_name} Predicted"] = pred_metrics
                else:
                    model_metrics[f"{model_name} True"] = np.where(np.isnan(valid_outcomes), np.nan, valid_outcomes)
                    model_metrics[f"{model_name} Predicted"] = np.where(np.isnan(valid_preds), np.nan, valid_preds)
                self.metrics[model_name] = model_metrics
                
            metrics.update(self.metrics[model_name])
                        
        return metrics
    
    def get_slice_ranking_info(self, timestep_def, controls, model_names):
        """Returns a RankedSliceList object with no slices, allowing clients to
        score custom slices without going through the slice search ranking process."""
        result_key = self.controls_to_result_key(controls)
        
        valid_df, eval_mask, ids, _ = self.get_slicing_data(controls["slice_spec_name"], timestep_def, True)
            
        score_fns, _ = self.get_score_functions(valid_df, eval_mask, timestep_def, controls=controls)
                
        metrics = self.get_eval_metrics(model_names, eval_mask)
        
        rank_list = sf.slices.RankedSliceList([], 
                                              valid_df, 
                                              score_fns, 
                                              similarity_threshold=0.7)
        rank_list.score_cache = self.eval_score_caches.setdefault(timestep_def, {}).setdefault(result_key, {})
        return rank_list, metrics, ids, valid_df

    def get_results(self, timestep_def, controls, model_names):
        current_status = self.get_status()
        if current_status != self.slice_finding_status:
            print("Refreshing stored slice scores")
            self.slice_finding_status = current_status
            
            # reload slice results
            self.load_timestep_slice_results(timestep_def)
        
        if (timestep_def not in self.slice_scores or not len(self.slice_scores[timestep_def])):
            return None
        
        result_key = self.controls_to_result_key(controls)
        print("result key for rescoring:", result_key)
        if result_key not in self.slice_scores.get(timestep_def, {}):
            return None
        else:
            scored_slices = list(self.slice_scores[timestep_def][result_key].keys())
        
        valid_df, eval_mask, ids, _ = self.get_slicing_data(controls["slice_spec_name"], timestep_def, True)
            
        score_fns, _ = self.get_score_functions(valid_df, eval_mask, timestep_def, controls=controls)
                
        metrics = self.get_eval_metrics(model_names, eval_mask)
        all_outcomes = np.vstack([self.get_model_labels(model_name)[eval_mask]
                                  for model_name in model_names])
        min_items_per_model = self.min_items_fraction * (~np.isnan(all_outcomes)).sum(axis=1) 
        scored_slices = [s for s in scored_slices if np.any((~np.isnan(all_outcomes[:,s.make_mask(valid_df.df)])).sum(axis=1) >= min_items_per_model, axis=0)]

        filters = []
        for i, model_name in enumerate(model_names):
            slice_filter = filter_single_values(valid_df, all_outcomes[i])
            if slice_filter: filters.append(slice_filter)
        if filters:
            overall_filter = sf.filters.ExcludeIfAll(filters)
            scored_slices = [s for s in scored_slices if overall_filter(s)]
        
        rank_list = sf.slices.RankedSliceList(scored_slices, 
                                              valid_df, 
                                              score_fns, 
                                              similarity_threshold=0.7)
        rank_list.score_cache = self.eval_score_caches.setdefault(timestep_def, {}).setdefault(result_key, {})
        return rank_list, metrics, ids, valid_df


def describe_slice_differences(variables, slice_obj, slice_filter=None, valid_mask=None, topk=50):
    """
    Generates a JSON-formatted description of the variables that have
    the greatest differences within the slice and on average.
    
    variables: A DiscretizedData object
    slice_obj: A slice within which to compare variable values, OR an array mask
        to apply directly to the variable matrix
    slice_filter: If provided, a slice filter object that can be called
        to determine if a feature value is allowed to be shown
    valid_mask: If provided, filter variables to only these rows
    """
    if isinstance(slice_obj, (sf.slices.SliceFeatureBase, sf.slices.Slice)):
        slice_mask = slice_obj.make_mask(variables.df).cpu().numpy()
        univariate_features = slice_obj.univariate_features()
    else:
        slice_mask = slice_obj
        univariate_features = None

    if valid_mask is None: valid_mask = np.ones(len(variables.df), dtype=bool)
    base_df = variables.df[valid_mask]
    slice_df = variables.df[slice_mask & valid_mask]

    variable_summaries = {}
    enrichment_scores = {}
    try:
        input_columns = base_df.columns
    except AttributeError:
        input_columns = np.arange(base_df.shape[1])
    for col in input_columns:
        if univariate_features is not None and any(f.feature_name == col for f in univariate_features):
            continue
        col_name, value_map = variables.value_names[col]
        for val, val_name in sorted(value_map.items(), key=lambda x: x[1] if x[1] != "End of Trajectory" else "zzzzz"):
            base_count = (base_df[:,col] == val).sum()
            slice_count = (slice_df[:,col] == val).sum()
            variable_summaries.setdefault(col_name, {"values": [], "base": [], "slice": []})
            variable_summaries[col_name]["values"].append(val_name)
            variable_summaries[col_name]["base"].append(base_count)
            variable_summaries[col_name]["slice"].append(slice_count)
            
            if slice_filter is not None and not slice_filter(sf.slices.SliceFeature(col, [val])): continue
            base_prob = base_count / len(base_df)
            slice_prob = slice_count / len(slice_df)
            enrichment_scores[(col_name, val_name)] = (base_prob, slice_prob, (1e-3 + slice_prob) / (1e-3 + base_prob))

    top_enrichments = sorted(enrichment_scores.items(), key=lambda x: x[1][-1], reverse=True)[:topk]
    top_variables = []
    for (col, val), scores in top_enrichments:
        existing = next((x for x in top_variables if x["variable"] == col), None)
        if existing is None:
            existing = {"variable": col, "enrichments": []}
            top_variables.append(existing)
        existing["enrichments"].append({"value": val, "ratio": (scores[1] - scores[0]) / scores[0]})
        
    return {"top_variables": top_variables, "all_variables": variable_summaries}

def describe_slice_change_differences(variables, ids, shift_steps, slice_obj, slice_filter=None, valid_mask=None, topk=50):
    """
    Generates a JSON-formatted description of the variables that change
    in the most different ways between the slice and the rest of the
    dataset.
    
    variables: A DiscretizedData object
    ids: A vector containing the IDs for each timestep in variables
    shift_steps: Number of steps to shift data forward (positive
        numbers compare to timesteps in the past, negative numbers
        compare to timesteps in the future)
    slice_obj: A slice within which to compare variable values
    slice_filter: If provided, a slice filter object that can be called
        to determine if a feature value is allowed to be shown
    valid_mask: If provided, filter variables to only these rows
    """
    source_description = describe_slice_differences(variables,
                                                      slice_obj,
                                                      slice_filter=slice_filter,
                                                      valid_mask=valid_mask,
                                                      topk=topk)
    
    shifted_values = sf.discretization.DiscretizedData((pd.DataFrame(variables.df)
                                                        .groupby(ids)
                                                        .shift(shift_steps) + 1).fillna(0).values,
                                                       {col: (col_name, {**{k + 1: v for k, v in value_map.items()},
                                                                         0: "End of Trajectory"})
                                                        for col, (col_name, value_map) in variables.value_names.items()})
    # Make sure to use the exact same slice mask as used in the source calculation
    slice_mask = slice_obj.make_mask(variables.df).cpu().numpy()
    # Decode and re-encode the slice filter since the value names have changed
    shifted_filter = shifted_values.encode_filter(variables.decode_filter(slice_filter))
    dest_description = describe_slice_differences(shifted_values,
                                                      slice_mask,
                                                      slice_filter=shifted_filter,
                                                      valid_mask=valid_mask,
                                                      topk=topk)
    
    # Now compute the changes
    univariate_features = slice_obj.univariate_features()

    if valid_mask is None: valid_mask = np.ones(len(variables.df), dtype=bool)
    base_df = variables.df[valid_mask]
    slice_df = variables.df[slice_mask & valid_mask]

    base_df_change = shifted_values.df[valid_mask]
    slice_df_change = shifted_values.df[slice_mask & valid_mask]

    change_scores = {}
    try:
        input_columns = base_df_change.columns
    except AttributeError:
        input_columns = np.arange(base_df_change.shape[1])

    for col in tqdm.tqdm(input_columns):
        if any(f.feature_name == col for f in univariate_features):
            continue
        col_name, value_map = variables.value_names[col]
        for val, val_name in value_map.items():
            if slice_filter is not None and not slice_filter(sf.slices.SliceFeature(col, [val])): continue
            base_prob_ref = (base_df[:,col] == val)
            slice_prob_ref = (slice_df[:,col] == val)
            for other_val, other_val_name in shifted_values.value_names[col][1].items():
                if slice_filter is not None and not slice_filter(sf.slices.SliceFeature(col, [other_val])): continue
                base_prob = (base_prob_ref & (base_df_change[:,col] == other_val)).mean()
                slice_prob = (slice_prob_ref & (slice_df_change[:,col] == other_val)).mean()
                change_scores[(col_name, val_name, other_val_name)] = (base_prob, slice_prob, (1e-3 + slice_prob) / (1e-3 + base_prob))

    top_enrichments = sorted(change_scores.items(), key=lambda x: x[1][-1], reverse=True)[:topk]
    top_variables = []
    for (col, val, other_val), scores in top_enrichments:
        existing = next((x for x in top_variables if x["variable"] == col), None)
        if existing is None:
            existing = {"variable": col, "enrichments": []}
            top_variables.append(existing)
        existing["enrichments"].append({"source_value": val,
                                        "destination_value": other_val, 
                                        "base_prob": scores[0],
                                        "slice_prob": scores[1],
                                        "ratio": (scores[1] - scores[0]) / scores[0]})
        
    
    return {"top_changes": top_variables, "source": source_description, "destination": dest_description}
