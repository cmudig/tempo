import pandas as pd
import os
import re
import json
import tqdm
import uuid
import pickle
import datetime
import time
import logging
import traceback
from ..query_language.data_types import *
from sklearn.metrics import roc_auc_score, confusion_matrix, r2_score, f1_score
from .utils import make_query
import divisi
from divisi.filters import ExcludeFeatureValueSet, ExcludeIfAny
from divisi.discretization import discretize_column, discretize_data, DiscretizedData
from divisi.slices import RankedSliceList, IntersectionSlice, SliceFeatureBase, Slice, SliceFeature
from divisi.utils import convert_to_native_types, powerset

class SlicingVariableSpec:
    def __init__(self, spec_fs):
        self.spec_fs = spec_fs
        self.discrete_df = None
        self.ids = None
        
    def get_spec(self):
        try:
            return self.spec_fs.read_file("spec.json")
        except:
            raise ValueError("Slicing variable spec does not exist")
    
    def write_spec(self, new_spec):
        self.spec_fs.write_file(new_spec, "spec.json")
    
    def create_default(self, model_spec, variables_df):
        """
        Creates an automatic default slicing variable spec based on the given
        model spec variables. Discrete and binary variables will be kept as-is,
        while continuous-valued variables will be transformed using quantile
        binning rounded to appropriate decimal places.
        """
        new_variables = {}
        for var_name, var_info in model_spec["variables"].items():
            if not var_info.get('enabled', True): continue
            
            var_values = variables_df[var_name]
            if (isinstance(var_values.dtype, pd.CategoricalDtype) or 
                pd.api.types.is_object_dtype(var_values.dtype) or 
                pd.api.types.is_string_dtype(var_values.dtype)):
                # already categorical data
                new_variables[var_name] = {"query": f"({var_info['query']}) impute 'Missing'"}
            else:
                uniques = var_values[~pd.isna(var_values)].unique()
                if len(uniques) == 2 and np.allclose(uniques, np.array([0, 1])):
                    # binary data
                    new_variables[var_name] = {"query": f"(case when ({var_info['query']}) then 'True' else 'False' end) impute 'Missing'"}
                elif len(uniques) <= 10:
                    # numeric data with a small number of categories
                    new_variables[var_name] = {"query": f"({var_info['query']}) impute 'Missing'"}
                else:
                    # numeric data
                    new_variables[var_name] = {"query": f"({var_info['query']}) cut quantiles [0, 0.1, 0.3, 0.7, 0.9, 1] impute 'Missing'"}
        self.write_spec({"variables": new_variables})
    
    def load_slicing_variables(self, query_engine, timestep_definition, update_fn=None):
        """Creates the slicing variables dataframe."""
        variable_definitions = self.get_spec()['variables']
        query = make_query(variable_definitions, timestep_definition)
        print("Generated query")
        
        # Save the discretization mapping to the evaluator cache file so we can recover it later
        discretization_results = {}
        def get_unique_values(var_exp):
            if var_exp.name:
                value_indexes, col_spec = discretize_column(var_exp.name, 
                                                            var_exp.get_values(), 
                                                            {"method": "unique"})
                discretization_results[var_exp.name] = col_spec
                return var_exp.with_values(pd.Series(value_indexes, name=var_exp.name)), col_spec
            return var_exp, None
        def restore_discretization_info(var_exp, info):
            # convert string keys to integers
            new_info = {}
            for k, v in info[1].items():
                try:
                    new_info[int(k)] = v
                except:
                    new_info[k] = v
            discretization_results[var_exp.name] = (var_exp.name, new_info)
            return var_exp
            
        if update_fn is not None:
            def prog(num_completed, num_total):
                update_fn({'message': f'Loading variables ({num_completed} / {num_total})', 'progress': num_completed / num_total})
        else:
            prog = None
        variable_df = query_engine.query(query, update_fn=prog, variable_transform=("unique_values", get_unique_values, restore_discretization_info))
        
        discrete_df = None
        if not discretization_results:
            print("Re-discretizing data because value names were not found")
            discrete_df = discretize_data(variable_df.values, {
                col: { "method": "keep" }
                for col in variable_df.values.columns
            })
        else:
            dataframe = variable_df.values
            value_names = {i: discretization_results[col] 
                        for i, col in enumerate(dataframe.columns)
                        if col in discretization_results}
            print("Value names:", value_names)
            # Find any keys that might not have been cached and re-discretize them
            missing_keys = [c for c in dataframe.columns if c not in discretization_results]
            print("Missing keys:", missing_keys)
            if missing_keys:
                processed_missing_df = discretize_data(dataframe[missing_keys], {
                    col: { "method": "unique"} for col in missing_keys
                })
                reverse_index = {c: i for i, c in enumerate(dataframe.columns)}
                value_names.update({reverse_index[missing_keys[i]]: v for i, v in  processed_missing_df.value_names.items()})
                dataframe = dataframe.assign(**{c: processed_missing_df.df[:,i] for i, c in enumerate(missing_keys)})
            print("Final value names:", value_names)
            discrete_df = DiscretizedData(dataframe.values.astype(np.uint8), value_names)
            
        print("Completed discretization")
        self.discrete_df = discrete_df
        self.ids = variable_df.index.get_ids()

class SliceFinder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.results_fs = self.dataset.global_cache_dir.subdirectory("slices", "results")
        self.variable_cache_fs = self.dataset.split_cache_dir.subdirectory("slicing_variables")
        self._results_cache = None # keys: strings of the format "model name###variable spec name"
        self._variable_specs_cache = {} # keys: tuples (timestep definition, variable spec name)
        self._model_cache = {} # keys: model name strings
        self._eval_metrics_cache = {} # keys: model name strings
        self._eval_slice_cache = {} # keys: timestep definitions
        
    def invalidate_model(self, model_name):
        self.load_cache_if_needed()
        self._results_cache = {k: v for k, v in self._results_cache.items()
                               if not re.match(f"^{re.escape(model_name)}###.*$", k)}
        self.write_cache()
        if model_name in self._model_cache: del self._model_cache[model_name]
        if model_name in self._eval_metrics_cache: del self._eval_metrics_cache[model_name]
    
    def invalidate_variable_spec(self, variable_spec_name):
        self.load_cache_if_needed()
        self._results_cache = {k: v for k, v in self._results_cache.items()
                               if not re.match(f"^.*###{re.escape(variable_spec_name)}$", k)}
        self.write_cache()
        self._variable_specs_cache = {k: v for k, v in self._variable_specs_cache.items()
                                      if k[1] != variable_spec_name}

    def _get_model(self, model_name):
        if model_name not in self._model_cache: self._model_cache[model_name] = self.dataset.get_model(model_name)
        return self._model_cache[model_name]
    
    def get_variable_spec(self, timestep_definition, variable_spec_name, load_if_needed=False, update_fn=None):
        if (timestep_definition, variable_spec_name) not in self._variable_specs_cache:
            self._variable_specs_cache[(timestep_definition, variable_spec_name)] = self.dataset.get_slicing_variable_spec(variable_spec_name)
        spec = self._variable_specs_cache[(timestep_definition, variable_spec_name)]
        if spec.discrete_df is None and load_if_needed:
            engine = self.dataset.make_query_engine(cache_fs=self.variable_cache_fs)
            spec.load_slicing_variables(engine, timestep_definition, update_fn=update_fn)
        return spec
        
    def filter_single_values(self, valid_df):
        # Exclude any features that have only one value
        single_value_filters = {}
        for col_idx, (col, value_pairs) in valid_df.value_names.items():
            unique_vals = np.unique(valid_df.df[:,col_idx])
            if len(unique_vals) <= 1:
                single_value_filters.setdefault(unique_vals[0], []).append(col_idx)
                
        print("Single value filters:", single_value_filters)
        if single_value_filters:
            return ExcludeIfAny([ExcludeFeatureValueSet(cols, [v]) for v, cols in single_value_filters.items()])

    def parse_score_expression(self, score_expression, split):
        assert "type" in score_expression, f"Score expression needs a 'type' field: {score_expression}"
        if score_expression["type"] == "model_property":
            model = self._get_model(score_expression["model_name"])
            if score_expression["property"] == "label":
                outcome = model.get_true_labels(split)
            elif score_expression["property"] == "prediction":
                try:
                    threshold = model.get_optimal_threshold()
                    outcome = model.get_model_predictions(split) >= threshold
                except:
                    outcome = model.get_model_predictions(split)
            elif score_expression["property"] == "prediction_probability":
                outcome = model.get_model_predictions(split)
            elif score_expression["property"] == "correctness":
                try:
                    threshold = model.get_optimal_threshold()
                    preds = model.get_model_predictions(split) >= threshold
                except:
                    preds = model.get_model_predictions(split)
                outcome = model.get_true_labels(split) == preds
            elif score_expression["property"] == "deviation":
                outcome = model.get_model_predictions(split) - model.get_true_labels(split)
            elif score_expression["property"] == "abs_deviation":
                outcome = np.abs(model.get_model_predictions(split) - model.get_true_labels(split))
            else:
                raise ValueError(f"Unknown score expression property '{score_expression['property']}'")
            return outcome
        elif score_expression["type"] == "constant":
            return score_expression["value"]
        elif score_expression["type"] in ("relation", "logical"):
            # "=" | "!=" | "<" | "<=" | ">" | ">=" | "in" | "not-in" | "and" | "or"
            assert 'lhs' in score_expression and 'rhs' in score_expression, f"Score expression relation needs both lhs and rhs: {score_expression}"
            lhs = self.parse_score_expression(score_expression["lhs"], split)
            rhs = self.parse_score_expression(score_expression["rhs"], split)
            if score_expression["relation"] == "=":
                return lhs == rhs
            elif score_expression["relation"] == "!=":
                return lhs != rhs
            elif score_expression["relation"] == "<":
                return lhs < rhs
            elif score_expression["relation"] == "<=":
                return lhs <= rhs
            elif score_expression["relation"] == ">":
                return lhs > rhs
            elif score_expression["relation"] == ">=":
                return lhs >= rhs
            elif score_expression["relation"] == "in":
                assert isinstance(rhs, (list, np.array)), f"Right-hand side of 'in' expression must be a list, but got {rhs}"
                return np.isin(lhs, rhs)
            elif score_expression["relation"] == "not-in":
                assert isinstance(rhs, (list, np.array)), f"Right-hand side of 'in' expression must be a list, but got {rhs}"
                return ~np.isin(lhs, rhs)
            elif score_expression["relation"] == "and":
                return np.logical_and(lhs, rhs)
            elif score_expression["relation"] == "or":
                return np.logical_or(lhs, rhs)
            else:
                raise ValueError(f"Unknown score expression relation '{score_expression['relation']}'")
        else:
            raise ValueError(f"Unknown score expression type '{score_expression['type']}'")
        
    def make_score_functions(self, score_function_spec, model_name=None):
        all_functions = ({}, {})
        all_metrics = ({}, {})
        sampling_mask = None
        for split, functions, metrics in zip(('val', 'test'), all_functions, all_metrics):
            for i, spec in enumerate(score_function_spec):
                score_fn_data = self.parse_score_expression(spec, split)
                print("For split", split, "score function data has", score_fn_data.mean())
                uniques = np.unique(score_fn_data).astype(int)
                assert len(uniques) <= 2 and not (set(uniques) - set([0, 1])), "Score functions must result in a binary value"
                functions[f"{i}"] = divisi.OutcomeRateScore(score_fn_data)
                functions[f"{i} Interaction"] = divisi.InteractionEffectScore(score_fn_data)
                metrics[f"{i}"] = score_fn_data
                if split == 'val':
                    if sampling_mask is None:
                        sampling_mask = (score_fn_data > 0)
                    else:
                        sampling_mask |= (score_fn_data > 0)
                
        all_masks = []
        if model_name is not None:
            for split in ('val', 'test'):
                valid_mask = ~np.isnan(self._get_model(model_name).get_true_labels(split))
                all_masks.append(valid_mask)
        
        weights = {n: 1.0 for n in all_functions[0].keys()}
        all_functions[0]["Large Slice"] = divisi.SliceSizeScore()
        all_functions[0]["Simple Rule"] = divisi.NumFeaturesScore()
        all_functions[1]["Large Slice"] = divisi.SliceSizeScore()
        all_functions[1]["Simple Rule"] = divisi.NumFeaturesScore()
        weights["Large Slice"] = 0.5
        weights["Simple Rule"] = 0.5
        return (all_functions, all_metrics, weights, tuple(all_masks), sampling_mask)
    
    def get_eval_metrics(self, model_names):
        """Returns a dictionary of model names and metric types (e.g. '{model name} True') to metric arrays
        over the evaluation dataset."""
        metrics = {}
        for model_name in model_names:
            if model_name not in self._eval_metrics_cache:
                model = self._get_model(model_name)
                spec = model.get_spec()
                if not model.is_trained:
                    raise ValueError(f"Model '{model_name}' is not trained or had an error, please retrain it.")                
                
                model_metrics = {}
                outcomes = model.get_true_labels('test')
                preds = model.get_model_predictions('test')
            
                valid_outcomes = outcomes.astype(np.float64)
                valid_preds = preds
                if len(valid_preds.shape) > 1: valid_preds = np.where(np.isnan(valid_outcomes), np.nan, np.argmax(valid_preds, axis=1))

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
                self._eval_metrics_cache[model_name] = model_metrics
                
            metrics.update(self._eval_metrics_cache[model_name])
                        
        return metrics

    
    def _find_slices(self, model_name, variable_spec, score_function_spec, update_fn=None, n_samples=100, n_slices=20, similarity_threshold=0.5, min_items_fraction=0.02, **kwargs):
        (discovery_score_fns, eval_score_fns), _, weights, valid_masks, sampling_mask = self.make_score_functions(score_function_spec, model_name)
        
        discovery_ids = self.dataset.split_ids[1]  # validation set
        discovery_df = variable_spec.discrete_df.filter(variable_spec.ids.isin(discovery_ids)).filter(valid_masks[0])
        discovery_filter = self.filter_single_values(discovery_df)
        
        if update_fn is not None: update_fn({"message": "Finding slices"})
        finder = divisi.sampling.SamplingSliceFinder(
            discovery_df, 
            discovery_score_fns,
            holdout_fraction=0.0,
            source_mask=sampling_mask,
            group_filter=discovery_filter,
            min_items=min_items_fraction * len(discovery_df),
            **kwargs
        )
        
        # if seen_slices is not None:
        #     finder.seen_slices = {**seen_slices}
        #     finder.all_scores += list(seen_slices.keys())
        
        def progress(current, total):
            if update_fn is not None: update_fn({ 'message': f'Finding slices {current} / {total}', 'progress': current / total})
        finder.progress_fn = progress
        results, _ = finder.sample(n_samples)
        
        if update_fn is not None: update_fn({"message": "Evaluating slices"})
        eval_ids = self.dataset.split_ids[2]
        eval_df = variable_spec.discrete_df.filter(variable_spec.ids.isin(eval_ids)).filter(valid_masks[1]) # validation set

        # Add superslices for each found slice
        slices_to_score = set(results.results)
        print("Before adding superslices:", len(slices_to_score))
        for slice_obj in results.results:
            univariate = slice_obj.univariate_features()
            for superslice_features in powerset(univariate):
                if len(superslice_features) == 0 or len(superslice_features) == len(univariate):
                    continue
                superslice = IntersectionSlice(superslice_features)
                if superslice in slices_to_score: continue
                slices_to_score.add(superslice)
        print("After adding superslices:", len(slices_to_score))
        
        scored_slices = divisi.slices.score_slices_batch(slices_to_score,
                                                         eval_df.df,
                                                         eval_score_fns,
                                                         finder.max_features,
                                                         min_items=min_items_fraction * len(eval_df))
        
        rank_list = RankedSliceList([v for v in scored_slices.values() if v], 
                                    eval_df, 
                                    eval_score_fns, 
                                    similarity_threshold=similarity_threshold)
        return rank_list.rank(weights, n_slices=n_slices)
        
    def load_cache_if_needed(self):
        try:    
            self._results_cache = self.results_fs.read_file("cache.json")
        except:
            self._results_cache = {}
                
    def lookup_slice_results(self, model_name, variable_spec_name, score_function_spec):
        """
        Returns the results of the slice finding operation (as Slice objects) 
        if they have already been computed. If the operation resulted in an error,
        raises that error as a ValueError."""
        self.load_cache_if_needed()
        
        key = f"{model_name}###{variable_spec_name}"
        if key not in self._results_cache:
            return None
        
        matching_result = next((result for result in self._results_cache[key]
                                 if result["score_function_spec"] == score_function_spec), None)
        if matching_result:
            try:
                results_json = self.results_fs.read_file(matching_result["path"])
            except:
                logging.info("Slice results file was removed")
                # file was removed
                self._results_cache[key] = [x for x in self._results_cache[key] if x != matching_result]
                self.write_cache()
            else:
                if isinstance(results_json, dict) and "error" in results_json:
                    raise ValueError(results_json["error"])
                
                results = [divisi.slices.Slice.from_dict(r) for r in results_json]
                return results
        return None

    def write_cache(self):
        self.results_fs.write_file(self._results_cache, "cache.json")
        
    def find_slices(self, model_name, variable_spec_name, score_function_spec, update_fn=None, ignore_cache=False, **options):
        """
        Runs the slice finding algorithm and returns an array of Slice objects.
        If an error occurred with the given slice finding operation, raises it
        as an Exception.
        """
        if not ignore_cache:
            cache_result = self.lookup_slice_results(model_name, variable_spec_name, score_function_spec)
            if cache_result: return cache_result
        
        key = f"{model_name}###{variable_spec_name}"
        path = uuid.uuid4().hex + ".json"
        
        # remove existing values for this score function spec
        self.load_cache_if_needed()
        if key in self._results_cache:
            for item in self._results_cache[key]:
                if item["score_function_spec"] == score_function_spec:
                    self.results_fs.delete(item["path"])
            self._results_cache[key] = [item for item in self._results_cache[key] 
                                        if item["score_function_spec"] != score_function_spec]

        try:
            # Check that the model was trained
            if not self._get_model(model_name).is_trained:
                raise ValueError(f"Model '{model_name}' is not trained or had an error, please retrain it.")
            
            if update_fn is not None: update_fn({"message": "Loading variables"})
            timestep_definition = self._get_model(model_name).get_spec()["timestep_definition"]
            variable_spec = self.get_variable_spec(timestep_definition, variable_spec_name, load_if_needed=True, update_fn=update_fn)
            
            results = self._find_slices(
                model_name,
                variable_spec,
                score_function_spec,
                update_fn=update_fn,
                **options
            )
            results_json = convert_to_native_types([slice_obj.to_dict() for slice_obj in results])
        except Exception as e:
            logging.info("Error loading model variables: " + traceback.format_exc())
            results_json = {"error": str(e)}
            
            self.results_fs.write_file(results_json, path)
            self._results_cache.setdefault(key, []).append({
                "score_function_spec": score_function_spec,
                "path": path
            })
            self.write_cache()
            raise e
        else:                    
            self.results_fs.write_file(results_json, path)
            self._results_cache.setdefault(key, []).append({
                "score_function_spec": score_function_spec,
                "path": path
            })
            self.write_cache()
        
        return results
    
    def evaluate_slices(self, slices, timestep_def, variable_spec_name, model_names, score_function_spec=None, include_meta=False, encode_slices=False):
        """
        Takes an array of Slice objects, and returns a dictionary of the format {
            "slices": [
                { slice description }, ...
            ],
            "base_slice": { slice description },
            "value_names": value names for discrete dataframe
        } if include_meta is True. Otherwise (default), returns just the value in
        the "slices" field above.
        """
        rank_list, metrics = self.get_slice_ranking_info(timestep_def, variable_spec_name, model_names)
        if score_function_spec is not None:
            score_metrics = self.make_score_functions(score_function_spec)[1][-1]
        else:
            score_metrics = None
        variable_spec = self.get_variable_spec(timestep_def, variable_spec_name, load_if_needed=True)
        ids = variable_spec.ids
        valid_df = variable_spec.discrete_df
        
        base_slice = divisi.slices.Slice(divisi.slices.SliceFeatureBase())
        if isinstance(slices, list):
            slices = [rank_list.encode_slice(v) for v in slices] if encode_slices else slices
            slice_descs = [self.describe_slice(rank_list, metrics, ids, slice_obj, model_names, score_metrics=score_metrics)
                       for slice_obj in slices]
        elif isinstance(slices, dict) and "type" not in slices:
            slices = {k: rank_list.encode_slice(v) for k, v in slices.items()} if encode_slices else slices
            slice_descs = {name: self.describe_slice(rank_list, metrics, ids, slice_obj, model_names, score_metrics=score_metrics)
                       for name, slice_obj in slices.items()}
            # make sure to preserve custom string representations
            for name, desc in slice_descs.items():
                desc["stringRep"] = name
        elif isinstance(slices, divisi.slices.Slice) or (encode_slices and isinstance(slices, dict)):
            slices = rank_list.encode_slice(slices) if encode_slices else slices
            slice_descs = self.describe_slice(rank_list, metrics, ids, slices, model_names, score_metrics=score_metrics)
        if include_meta:
            return {
                "slices": slice_descs,
                "base_slice": self.describe_slice(rank_list, metrics, ids, base_slice, model_names, score_metrics=score_metrics),
                "value_names": valid_df.value_names
            }
        return slice_descs
        
    def get_slice_ranking_info(self, timestep_def, variable_spec_name, model_names):
        """Returns a RankedSliceList object with no slices, allowing clients to
        score custom slices without going through the slice search ranking process."""
        
        metrics = self.get_eval_metrics(model_names)
        
        variable_spec = self.get_variable_spec(timestep_def, variable_spec_name, load_if_needed=True)
        
        rank_list = divisi.slices.RankedSliceList([], variable_spec.discrete_df, {})
        rank_list.score_cache = self._eval_slice_cache.setdefault(timestep_def, {})
        return rank_list, metrics

    def describe_slice(self, rank_list, metrics, ids, slice_obj, model_names, return_instance_info=False, score_metrics=None):
        """
        Generates a slice description of the given slice, adding count variables
        for each model outcome.
        """
        desc, mask = rank_list.generate_slice_description(slice_obj, metrics=metrics, return_slice_mask=True)
        old_desc_metrics = desc["metrics"]
        desc["metrics"] = {}
        
        if score_metrics is not None:
            score_metrics_desc = rank_list.generate_slice_description(slice_obj, metrics=score_metrics)
            desc["metrics"]["Search Criteria"] = {k: v for k, v in score_metrics_desc["metrics"].items() if k != "Count"}
        
        instance_behaviors = {}
        for model_name in model_names:
            model = self.dataset.get_model(model_name)
            spec = model.get_spec()
            
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
            if matching_nonna == 0:
                continue
            desc["metrics"][model_name].update({
                "Labels": old_desc_metrics[f"{model_name} True"],
                "Predictions": old_desc_metrics[f"{model_name} Predicted"]
            })
            
            if spec["model_type"] == "binary_classification":
                opt_threshold = model.get_optimal_threshold()
                    
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

    def describe_slice_differences(self, slice_dict, model_name, variable_spec_name, variables=None, slice_filter=None, topk=50):
        """
        Generates a JSON-formatted description of the variables that have
        the greatest differences within the slice and on average.
        
        slice_obj: A slice within which to compare variable values, OR an array mask
            to apply directly to the variable matrix
        slice_filter: If provided, a slice filter object that can be called
            to determine if a feature value is allowed to be shown
        valid_mask: If provided, filter variables to only these rows
        """
        if variables is None:
            # load only test set
            timestep_definition = self._get_model(model_name).get_spec()["timestep_definition"]
            variable_spec = self.get_variable_spec(timestep_definition, variable_spec_name, load_if_needed=True)
            eval_ids = self.dataset.split_ids[2]
            valid_mask = ~np.isnan(self._get_model(model_name).get_true_labels('test'))
            variables = variable_spec.discrete_df.filter(variable_spec.ids.isin(eval_ids)).filter(valid_mask)
        slice_obj = variables.encode_slice(slice_dict)
            
        slice_mask = slice_obj.make_mask(variables.df).cpu().numpy()
        univariate_features = slice_obj.univariate_features()

        base_df = variables.df
        slice_df = variables.df[slice_mask]

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
                
                if slice_filter is not None and not slice_filter(SliceFeature(col, [val])): continue
                base_prob = base_count / len(base_df)
                slice_prob = slice_count / len(slice_df)
                if slice_prob < base_prob: continue
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

    def describe_slice_change_differences(self, shift_steps, slice_dict, model_name, variable_spec_name, slice_filter=None, topk=50):
        """
        Generates a JSON-formatted description of the variables that change
        in the most different ways between the slice and the rest of the
        dataset.
        
        shift_steps: Number of steps to shift data forward (positive
            numbers compare to timesteps in the past, negative numbers
            compare to timesteps in the future)
        slice_obj: A slice within which to compare variable values
        slice_filter: If provided, a slice filter object that can be called
            to determine if a feature value is allowed to be shown
        """
        timestep_definition = self._get_model(model_name).get_spec()["timestep_definition"]
        variable_spec = self.get_variable_spec(timestep_definition, variable_spec_name, load_if_needed=True)
        eval_ids = self.dataset.split_ids[2]
        valid_mask = ~np.isnan(self._get_model(model_name).get_true_labels('test'))
        variables = variable_spec.discrete_df.filter(variable_spec.ids.isin(eval_ids)).filter(valid_mask)
        ids = variable_spec.ids[variable_spec.ids.isin(eval_ids)]
        slice_obj = variables.encode_slice(slice_dict)
        
        source_description = self.describe_slice_differences(slice_dict,
                                                             model_name,
                                                             variable_spec_name,
                                                             variables=variables,
                                                             slice_filter=slice_filter,
                                                             topk=topk)
        
        shifted_values = DiscretizedData((pd.DataFrame(variables.df)
                                                            .groupby(ids)
                                                            .shift(shift_steps) + 1).fillna(0).values,
                                                        {col: (col_name, {**{k + 1: v for k, v in value_map.items()},
                                                                            0: "End of Trajectory"})
                                                            for col, (col_name, value_map) in variables.value_names.items()})
        # Make sure to use the exact same slice mask as used in the source calculation
        slice_mask = slice_obj.make_mask(variables.df).cpu().numpy()
        # Decode and re-encode the slice filter since the value names have changed
        if slice_filter is not None:
            shifted_filter = shifted_values.encode_filter(variables.decode_filter(slice_filter))
        else:
            shifted_filter = None
        dest_description = self.describe_slice_differences(slice_dict,
                                                           model_name,
                                                           variable_spec_name,
                                                        variables=shifted_values,
                                                        slice_filter=shifted_filter,
                                                        topk=topk)
        
        # Now compute the changes
        univariate_features = slice_obj.univariate_features()

        base_df = variables.df
        slice_df = variables.df[slice_mask]

        base_df_change = shifted_values.df
        slice_df_change = shifted_values.df[slice_mask]

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
                if slice_filter is not None and not slice_filter(SliceFeature(col, [val])): continue
                base_prob_ref = (base_df[:,col] == val)
                slice_prob_ref = (slice_df[:,col] == val)
                for other_val, other_val_name in shifted_values.value_names[col][1].items():
                    if slice_filter is not None and not slice_filter(SliceFeature(col, [other_val])): continue
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
