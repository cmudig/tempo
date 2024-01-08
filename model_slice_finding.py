import pandas as pd
import os
import re
import json
import torch
import pickle
import datetime
from query_language.data_types import *
from model_training import make_query, MODEL_DIR, load_raw_data
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import slice_finding as sf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def make_slicing_variables(dataset, variable_definitions, timestep_definition):
    """Creates the slicing variables dataframe."""
    query = make_query(variable_definitions, timestep_definition)
    print(query)
    variable_df = dataset.query(query)
    
    discrete_df = sf.discretization.discretize_data(variable_df.values, {
        col: { "method": "unique" }
        for col in variable_df.values.columns
    })
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
            new_score_fns["contains_slice"] = sf.scores.SliceSimilarityScore(ref_mask, metric='superslice')
            exclusion_criteria = sf.filters.ExcludeIfAny([
                sf.filters.ExcludeFeatureValueSet([f.feature_name], f.allowed_values)
                for f in contains_slice.univariate_features()
            ])
            new_source_mask &= ref_mask

    if (contained_in_slice := controls.get("contained_in_slice", {})) != {}:
        contained_in_slice = discrete_df.encode_slice(contained_in_slice)
        if contained_in_slice.feature != sf.slices.SliceFeatureBase():
            ref_mask = contained_in_slice.make_mask(raw_inputs).cpu().numpy()
            new_score_fns["contained_in_slice"] = sf.scores.SliceSimilarityScore(ref_mask, metric='subslice')
            exclusion_criteria = sf.filters.ExcludeIfAny([
                sf.filters.ExcludeFeatureValueSet([f.feature_name], f.allowed_values)
                for f in contained_in_slice.univariate_features()
            ])
            new_source_mask &= ref_mask

    if (similar_to_slice := controls.get("similar_to_slice", {})) != {}:
        similar_to_slice = discrete_df.encode_slice(similar_to_slice)
        if similar_to_slice.feature != sf.slices.SliceFeatureBase():
            ref_mask = similar_to_slice.make_mask(raw_inputs).cpu().numpy()
            new_score_fns["similar_to_slice"] = sf.scores.SliceSimilarityScore(ref_mask, metric='jaccard')
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
        weights["contains_slice"] = 1.0

    if controls.get("contained_in_slice", {}) != {}:
        weights["contained_in_slice"] = 1.0

    if controls.get("similar_to_slice", {}) != {}:
        weights["similar_to_slice"] = 1.0

    return weights        
    
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
            source_mask=finder.source_mask & new_source_mask,
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
    def __init__(self, model_dir, results_dir):
        self.model_dir = model_dir
        self.results_dir = results_dir        
        
    def get_status(self):
        status_path = os.path.join(self.results_dir, "discovery_status.json")
        try:
            if os.path.exists(status_path):
                with open(status_path, "r") as file:
                    return json.load(file)
        except:
            pass
        return {"searching": False, "status": {"state": "none", "message": "Not currently finding slices"}, "n_results": 0, "n_runs": 0, "models": []}
        
    def get_score_functions(self, discrete_df, valid_mask, include_model_names=None, exclude_model_names=None, controls=None):
        score_fns = {
            "size": sf.scores.SliceSizeScore(0.2, 0.05),
            "complexity": sf.scores.NumFeaturesScore()
        }
        model_names = []
        for path in os.listdir(self.model_dir):
            if not path.startswith("preds"): continue
            model_name = re.search(r"^preds_(.*).npy$", path).group(1)
            if include_model_names is not None and model_name not in include_model_names: continue
            if exclude_model_names is not None and model_name in exclude_model_names: continue
            
            model_names.append(model_name)
            
            outcomes, preds = np.load(os.path.join(self.model_dir, f"preds_{model_name}.npy"), allow_pickle=True).T.astype(float)
            with open(os.path.join(self.model_dir, f"metrics_{model_name}.json"), "r") as file:
                metrics = json.load(file)
                threshold = metrics['threshold']
                preds = np.where(np.isnan(preds), np.nan, preds >= threshold)
                
            valid_outcomes = outcomes[valid_mask].astype(np.float64)
            valid_preds = preds[valid_mask]
            valid_true = np.where(np.isnan(valid_outcomes), np.nan, valid_outcomes > 0)
            score_fns.update({
                f"{model_name}_true": sf.scores.OutcomeRateScore(valid_true),
                f"{model_name}_true_share": sf.scores.OutcomeShareScore(valid_true),
                f"{model_name}_true_xf": sf.scores.InteractionEffectScore(valid_true),
                f"{model_name}_pred": sf.scores.OutcomeRateScore(valid_preds),
                f"{model_name}_pred_share": sf.scores.OutcomeShareScore(valid_preds),
                f"{model_name}_pred_xf": sf.scores.InteractionEffectScore(valid_preds),
            })
            
        if controls is not None:
            new_score_fns, _, _, _ = parse_controls(discrete_df, controls)
            score_fns.update(new_score_fns)
        return score_fns, model_names

    def get_model_timestep_def(self, model_name):
        with open(os.path.join(self.model_dir, f"spec_{model_name}.json"), "r") as file:
            spec = json.load(file)
        return spec['timestep_definition']
    
    def controls_to_result_key(self, controls):
        """Returns a key into the slice results file representing the results for
        this set of controls."""
        def make_slice(obj):
            if obj is None: return None
            return sf.slices.SliceFeatureBase.from_dict(obj)
        return (make_slice(controls.get("contains_slice", None)),
                make_slice(controls.get("contained_in_slice", None)),
                make_slice(controls.get("similar_to_slice", None)),
                make_slice(controls.get("subslice_of_slice", None)))
        
    def result_key_to_controls(self, result_key):
        """Returns a key into the slice results file representing the results for
        this set of controls."""
        return {"contains_slice": result_key[0].to_dict() if result_key[0] is not None else None,
                "contained_in_slice": result_key[1].to_dict() if result_key[1] is not None else None,
                "similar_to_slice": result_key[2].to_dict() if result_key[2] is not None else None,
                "subslice_of_slice": result_key[3].to_dict() if result_key[3] is not None else None}
    
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
    def __init__(self, model_dir, results_dir, samples_per_model=50, min_items_fraction=0.01, **slice_finding_kwargs):
        super().__init__(model_dir, results_dir)
        self.samples_per_model = samples_per_model
        self.min_items_fraction = min_items_fraction
        self.discrete_dfs = {}
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
        
    def get_slicing_variables(self, timestep_def):
        if timestep_def not in self.discrete_dfs:
            dataset, _ = load_raw_data(cache_dir=os.path.join(DATA_DIR, "slicing_variables"), val_only=True)
            
            with open(os.path.join(DATA_DIR, "slice_variables.json"), "r") as file:
                discrete_df, _ = make_slicing_variables(dataset, json.load(file), timestep_def) 
                discovery_mask = get_slicing_split(self.results_dir, discrete_df, timestep_def)[0]
                valid_df = discrete_df.filter(discovery_mask)
                self.discrete_dfs[timestep_def] = valid_df
        else:
            valid_df = self.discrete_dfs[timestep_def]
            discovery_mask = get_slicing_split(self.results_dir, valid_df, timestep_def)[0]
        return valid_df, discovery_mask 
        
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
        
        valid_df, discovery_mask = self.get_slicing_variables(timestep_def)
        
        for _, control_results in self.slice_scores[timestep_def].items():
            other_score_fns, scored_model_names = self.get_score_functions(valid_df, discovery_mask, include_model_names=[model_name])
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
                if r in self.slice_scores[timestep_def]:
                    del self.slice_scores[timestep_def][r]
                control_results[r] = r.score_values
        
        print("Writing to file")
        self.save_timestep_slice_results(timestep_def)
        self.write_status(False, timestep_defs_updated=[timestep_def])
        print("Done")
        
    def find_slices(self, model_name, controls, kwarg_function=None):
        try:
            self.write_status(True, search_status={"state": "loading", "message": "Loading data", "model_name": model_name})
            
            timestep_def = self.get_model_timestep_def(model_name)
            self.load_timestep_slice_results(timestep_def)
            self.write_status(True, search_status={"state": "loading", "message": "Loading variables", "model_name": model_name})
            valid_df, discovery_mask = self.get_slicing_variables(timestep_def)
                 
            self.write_status(True, search_status={"state": "loading", "message": f"Finding slices for {model_name}", "model_name": model_name})
            
            outcomes, _ = np.load(os.path.join(MODEL_DIR, f"preds_{model_name}.npy"), allow_pickle=True).T.astype(float)
            # don't add control-related score functions here as they will be added
            # in find_slices
            discovery_score_fns, _ = self.get_score_functions(valid_df, discovery_mask, include_model_names=[model_name])
            
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
                                  search_status={"state": "loading", "message": f"Finding slices for {model_name} ({progress} / {total})", "progress": progress / total, "model_name": model_name},
                                  new_runs=progress - last_progress)
                last_progress = progress
            
            print(len(discovery_outcomes), "outcomes")
            min_items = self.min_items_fraction * len(discovery_df.df)
            self.slice_scores.setdefault(timestep_def, {})
            command_results = self.slice_scores[timestep_def].setdefault(result_key, {})
            print(discovery_df.df.shape, min_items, self.slice_finding_kwargs)

            results = find_slices(discovery_df, 
                                    discovery_score_fns,
                                    controls=controls,
                                    progress_fn=update_sampler_progress, 
                                    seen_slices=command_results,
                                    n_samples=self.samples_per_model, 
                                    min_items=min_items,
                                    n_workers=None,
                                    source_mask=discovery_outcomes > 0,
                                    **self.slice_finding_kwargs,
                                    **(kwarg_function(discovery_df, discovery_outcomes) if kwarg_function is not None else {}))

            # Add scores for all the other models
            other_score_fns, scored_model_names = self.get_score_functions(valid_df, discovery_mask)
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
            
            print(self.slice_scores[timestep_def].keys(), [len(self.slice_scores[timestep_def][r]) for r in self.slice_scores[timestep_def]])
            self.save_timestep_slice_results(timestep_def)
            self.write_status(False, timestep_defs_updated=[timestep_def], new_results=len(results), models_to_add=list(set([model_name, *scored_model_names])))
        except KeyboardInterrupt:
            self.write_status(False, search_status=None)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.write_status(False, error_model=model_name, error_message=str(e))
            
class SliceEvaluationHelper(SliceHelper):
    def __init__(self, model_dir, results_dir, **slice_finding_kwargs):
        super().__init__(model_dir, results_dir)
        self.slice_finding_kwargs = slice_finding_kwargs
        self.slice_finding_status = None
        self.slice_scores = {}
        self.discrete_dfs = {}
        self.eval_score_caches = {}
        self.metrics = {}
        self.eval_ids = {} 
        
    def get_default_weights(self, model_names, controls=None):
        if controls is not None:
            control_weights = default_control_weights(controls)
        else:
            control_weights = {}
        return {**{n: 1.0 for model_name in model_names
                   for n in [f"{model_name}_true", f"{model_name}_true_share", f"{model_name}_true_xf",
                             f"{model_name}_pred", f"{model_name}_pred_share", f"{model_name}_pred_xf"]},
                **control_weights,
                "size": 0.5, "complexity": 0.5}
        
    def describe_slice(self, rank_list, metrics, ids, slice_obj, model_names):
        """
        Generates a slice description of the given slice, adding count variables
        for each model outcome.
        """
        desc, mask = rank_list.generate_slice_description(slice_obj, metrics=metrics, return_slice_mask=True)
        old_desc_metrics = desc["metrics"]
        desc["metrics"] = {}
        
        for model_name in model_names:
            true_outcome = metrics[f"{model_name} True"]
            total_nonna = (~pd.isna(true_outcome)).sum()
            matching_nonna = (~pd.isna(true_outcome[mask])).sum()
            
            total_unique_ids = len(np.unique(ids[~pd.isna(true_outcome)]))
            
            pred_outcome = metrics[f"{model_name} Predicted"]
            slice_true = true_outcome[mask]
            slice_pred = pred_outcome[mask]
            slice_ids = ids[mask][~pd.isna(slice_true)]
            slice_pred = slice_pred[~pd.isna(slice_true)]
            slice_true = slice_true[~pd.isna(slice_true)]
            fpr, tpr, thresholds = roc_curve(slice_true, slice_pred)
            opt_threshold = thresholds[np.argmax(tpr - fpr)]
            auroc = roc_auc_score(slice_true, slice_pred)
            conf = confusion_matrix(slice_true, (slice_pred >= opt_threshold))
            tn, fp, fn, tp = conf.ravel()
            
            acc = ((slice_pred >= opt_threshold) == slice_true).mean()
            matching_unique_ids = len(np.unique(slice_ids))
            
            desc["metrics"][model_name] = {
                "Timesteps": {"type": "count", "count": matching_nonna, "share": matching_nonna / total_nonna},
                "Trajectories": {"type": "count", 
                                 "count": matching_unique_ids, 
                                 "share": matching_unique_ids / total_unique_ids},
                "Positive Rate": old_desc_metrics[f"{model_name} True"],
                "Accuracy": {"type": "binary", 
                             "count": matching_nonna, 
                             "mean": acc},
                "AUROC": {"type": "binary", 
                             "count": matching_nonna, 
                             "mean": auroc},
                "Sensitivity": {"type": "binary", 
                             "count": matching_nonna, 
                             "mean": float(tp / (tp + fn))},
                "Specificity": {"type": "binary", 
                             "count": matching_nonna, 
                             "mean": float(tn / (tn + fp))},
            }
            
        # Remove scores that aren't related to these model names
        desc["scoreValues"] = {k: v for k, v in desc["scoreValues"].items() 
                               if k in ("size", "complexity") or any(k.startswith(model_name + "_") for model_name in model_names)}
        return desc
        
    def rescore_model(self, model_name, timestep_def=None):
        """Recalculates the evaluation slice scores for the given model."""
        print("Rescoring model")
        if model_name in self.metrics: 
            del self.metrics[model_name]
            
        # Clear slice score cache so that the slices will be rescored
        if timestep_def is None: timestep_def = self.get_model_timestep_def(model_name)
        self.eval_score_caches[timestep_def] = {}
        
    def get_results(self, timestep_def, controls, model_names):
        current_status = self.get_status()
        if current_status != self.slice_finding_status:
            print("Refreshing stored slice scores")
            self.slice_finding_status = current_status
            
            # reload slice results
            self.load_timestep_slice_results(timestep_def)
        
        if timestep_def not in self.slice_scores or not len(self.slice_scores[timestep_def]):
            return None
        
        result_key = self.controls_to_result_key(controls)
        print("result key for rescoring:", result_key)
        if result_key not in self.slice_scores[timestep_def]:
            return None
        scored_slices = list(self.slice_scores[timestep_def][result_key].keys())
        
        if timestep_def not in self.discrete_dfs:
            dataset, _ = load_raw_data(cache_dir=os.path.join(DATA_DIR, "slicing_variables"), val_only=True)
            
            with open(os.path.join(DATA_DIR, "slice_variables.json"), "r") as file:
                discrete_df, ids = make_slicing_variables(dataset, json.load(file), timestep_def) 
                eval_mask = get_slicing_split(self.results_dir, discrete_df, timestep_def)[1]
                valid_df = discrete_df.filter(eval_mask)
                ids = ids.values[eval_mask]
                self.discrete_dfs[timestep_def] = valid_df
                self.eval_ids[timestep_def] = ids
        else:
            valid_df = self.discrete_dfs[timestep_def]
            eval_mask = get_slicing_split(self.results_dir, valid_df, timestep_def)[1]
            ids = self.eval_ids[timestep_def]
            
        score_fns, _ = self.get_score_functions(valid_df, eval_mask, controls=controls)
        
        metrics = {}
        for model_name in model_names:
            if model_name not in self.metrics:
                model_metrics = {}
                outcomes, preds = np.load(os.path.join(MODEL_DIR, f"preds_{model_name}.npy"), allow_pickle=True).T.astype(float)
            
                valid_outcomes = outcomes[eval_mask].astype(np.float64)
                valid_preds = preds[eval_mask]

                model_metrics[f"{model_name} True"] = np.where(np.isnan(valid_outcomes), np.nan, valid_outcomes > 0)
                model_metrics[f"{model_name} Predicted"] = np.where(np.isnan(valid_preds), np.nan, valid_preds)
                self.metrics[model_name] = model_metrics
                
            metrics.update(self.metrics[model_name])
        
        rank_list = sf.slices.RankedSliceList(scored_slices, 
                                              valid_df, 
                                              score_fns, 
                                              similarity_threshold=0.7)
        rank_list.score_cache = self.eval_score_caches.setdefault(timestep_def, {}).setdefault(result_key, {})
        return rank_list, metrics, ids, valid_df
