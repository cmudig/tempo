import pandas as pd
import os
import re
import json
import pickle
from query_language.data_types import *
from model_training import make_query, MODEL_DIR, load_raw_data
from sklearn.metrics import roc_curve
import slice_finding as sf
  
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
        discovery_mask = (np.random.uniform(size=dataset.df.shape[0]) >= HOLDOUT_FRACTION)
        eval_mask = ~discovery_mask
        splits[timestep_definition] = (discovery_mask, eval_mask)
        with open(os.path.join(slices_dir, "slicing_split.pkl"), "wb") as file:
            pickle.dump(splits, file)
    
    return splits[timestep_definition]
    
def find_slices(discrete_df, score_fns, progress_fn=None, n_samples=100, seen_slices=None, **kwargs):
    """
    Returns a tuple containing (metrics dict, slices list).
    """
    
    finder = sf.sampling.SamplingSliceFinder(
        discrete_df, 
        score_fns,
        holdout_fraction=0.0,
        **kwargs
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
        
    def get_score_functions(self, valid_mask, include_model_names=None, exclude_model_names=None):
        score_fns = {
            "size": sf.scores.SliceSizeScore(0.2, 0.05),
            "complexity": sf.scores.NumFeaturesScore()
        }
        for path in os.listdir(self.model_dir):
            if not path.startswith("preds"): continue
            model_name = re.search(r"^preds_(.*).npy$", path).group(1)
            if include_model_names is not None and model_name not in include_model_names: continue
            if exclude_model_names is not None and model_name in exclude_model_names: continue
            
            outcomes, preds = np.load(os.path.join(MODEL_DIR, f"preds_{model_name}.npy"), allow_pickle=True).T.astype(float)
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
        return score_fns
        
    
class SliceDiscoveryHelper(SliceHelper):
    def __init__(self, model_dir, results_dir, samples_per_model=50, min_items_fraction=0.01, **slice_finding_kwargs):
        super().__init__(model_dir, results_dir)
        self.samples_per_model = samples_per_model
        self.min_items_fraction = min_items_fraction
        self.discrete_dfs = {}
        self.slice_finding_kwargs = slice_finding_kwargs
        self.slice_scores = {}
        
    def write_status(self, searching, search_status=None, new_results=0, new_runs=0, models_to_add=None, models_to_remove=None, error_model=None, error_message=None):
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
        status_path = os.path.join(self.results_dir, "discovery_status.json")
        with open(status_path, "w") as file:
            json.dump(status, file)
            
    def invalidate_model(self, model_name):
        """Recalculates the discovery set scores for the given model."""
        raise NotImplementedError
        
    def find_slices(self, model_name, kwarg_function=None):
        try:
            self.write_status(True, search_status={"state": "loading", "message": "Loading data", "model_name": model_name})
            
            with open(os.path.join(self.model_dir, f"spec_{model_name}.json"), "r") as file:
                spec = json.load(file)
                
            results_path = os.path.join(self.results_dir, f"slice_results_{spec['timestep_definition']}.json")
            timestep_def = spec['timestep_definition']
            if os.path.exists(results_path):
                with open(results_path, "r") as file:
                    self.slice_scores[timestep_def] = {s: s.score_values for s in (sf.slices.Slice.from_dict(r) for r in json.load(file)["results"])}
                    
            if timestep_def not in self.discrete_dfs:
                dataset, _ = load_raw_data(cache_dir="data/slicing_variables", val_only=True)
                self.write_status(True, search_status={"state": "loading", "message": "Loading variables", "model_name": model_name})
                
                with open(os.path.join("data", "slice_variables.json"), "r") as file:
                    discrete_df, _ = make_slicing_variables(dataset, json.load(file), timestep_def) 
                    discovery_mask = get_slicing_split(self.results_dir, discrete_df, timestep_def)[0]
                    valid_df = discrete_df.filter(discovery_mask)
                    self.discrete_dfs[timestep_def] = valid_df
            else:
                valid_df = self.discrete_dfs[timestep_def]
                discovery_mask = get_slicing_split(self.results_dir, valid_df, timestep_def)[0]
                    
            self.write_status(True, search_status={"state": "loading", "message": f"Finding slices for {model_name}", "model_name": model_name})
            
            outcomes, _ = np.load(os.path.join(MODEL_DIR, f"preds_{model_name}.npy"), allow_pickle=True).T.astype(float)
            discovery_score_fns = self.get_score_functions(discovery_mask, include_model_names=[model_name])
            
            last_progress = 0
            
            def update_sampler_progress(progress, total):
                nonlocal last_progress
                self.write_status(True, 
                                  search_status={"state": "loading", "message": f"Finding slices for {model_name} ({progress} / {total})", "progress": progress / total, "model_name": model_name},
                                  new_runs=progress - last_progress)
                last_progress = progress
            
            print(len(outcomes[discovery_mask]), "outcomes")
            min_items = self.min_items_fraction * len(valid_df.df)
            self.slice_scores.setdefault(timestep_def, {})
            results = find_slices(valid_df, 
                                    discovery_score_fns,
                                    progress_fn=update_sampler_progress, 
                                    seen_slices=self.slice_scores[timestep_def],
                                    n_samples=self.samples_per_model, 
                                    min_items=min_items,
                                    n_workers=None,
                                    source_mask=outcomes[discovery_mask] > 0,
                                    **self.slice_finding_kwargs,
                                    **(kwarg_function(valid_df) if kwarg_function is not None else {}))

            # Add scores for all the other models
            other_score_fns = self.get_score_functions(discovery_mask, exclude_model_names=[model_name])
            new_results = [r for r in results if r not in self.slice_scores[timestep_def]]
            rescored_results = sf.slices.score_slices_batch(new_results, 
                                                            valid_df.df, 
                                                            other_score_fns, 
                                                            self.slice_finding_kwargs.get("max_features", 3), 
                                                            min_items=min_items, 
                                                            device='cpu')
            
            for r in new_results:
                r = r.rescore({**r.score_values, **(rescored_results[r].score_values if rescored_results.get(r, None) is not None else {})})
                if r in self.slice_scores[timestep_def]:
                    del self.slice_scores[timestep_def][r]
                self.slice_scores[timestep_def][r] = r.score_values
            
            with open(results_path, "w") as file:
                json.dump(sf.utils.convert_to_native_types({"results": [r.to_dict() for r in self.slice_scores[timestep_def]]}), file)
            self.write_status(False, new_results=len(results), models_to_add=[model_name])
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
        
    def get_default_weights(self, model_names):
        return {**{n: 1.0 for model_name in model_names
                   for n in [f"{model_name}_true", f"{model_name}_true_share", f"{model_name}_true_xf",
                             f"{model_name}_pred", f"{model_name}_pred_share", f"{model_name}_pred_xf"]},
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
            fpr, tpr, thresholds = roc_curve(slice_true[~pd.isna(slice_true)], slice_pred[~pd.isna(slice_true)])
            opt_threshold = thresholds[np.argmax(tpr - fpr)]
            acc = ((slice_pred[~pd.isna(slice_true)] >= opt_threshold) == slice_true[~pd.isna(slice_true)]).mean()
            matching_unique_ids = len(np.unique(ids[mask][~pd.isna(slice_true)]))
            
            print(desc["feature"], model_name, total_nonna, matching_nonna)
            desc["metrics"][model_name] = {
                "Timesteps": {"type": "count", "count": matching_nonna, "share": matching_nonna / total_nonna},
                "Trajectories": {"type": "count", 
                                 "count": matching_unique_ids, 
                                 "share": matching_unique_ids / total_unique_ids},
                "True": old_desc_metrics[f"{model_name} True"],
                "Accuracy": {"type": "binary", 
                             "count": matching_nonna, 
                             "mean": acc}
            }
            
        # Remove scores that aren't related to these model names
        desc["scoreValues"] = {k: v for k, v in desc["scoreValues"].items() 
                               if k in ("size", "complexity") or any(k.startswith(model_name + "_") for model_name in model_names)}
        return desc
        
    def get_results(self, timestep_def, model_names):
        current_status = self.get_status()
        if current_status != self.slice_finding_status:
            self.slice_finding_status = current_status
            
            # reload slice results
            results_path = os.path.join(self.results_dir, f"slice_results_{timestep_def}.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as file:
                    self.slice_scores[timestep_def] = [sf.slices.Slice.from_dict(r) for r in json.load(file)["results"]]
        
        if timestep_def not in self.slice_scores or not len(self.slice_scores[timestep_def]):
            return None
        
        scored_slices = self.slice_scores[timestep_def]
        
        if timestep_def not in self.discrete_dfs:
            dataset, _ = load_raw_data(cache_dir="data/slicing_variables", val_only=True)
            
            with open(os.path.join("data", "slice_variables.json"), "r") as file:
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
            
        score_fns = self.get_score_functions(eval_mask)
        
        metrics = {}
        for model_name in model_names:
            if model_name not in self.metrics:
                model_metrics = {}
                outcomes, preds = np.load(os.path.join(MODEL_DIR, f"preds_{model_name}.npy"), allow_pickle=True).T.astype(float)
            
                valid_outcomes = outcomes[eval_mask].astype(np.float64)
                valid_preds = preds[eval_mask]
                test_slice = valid_df.encode_slice(sf.slices.SliceFeatureNegation(
                    sf.slices.SliceFeature("Phenylephrine", ["None"]),
                ).to_dict())
                test_mask = test_slice.make_mask(valid_df.df)
                print(model_name)
                print(pd.isna(np.where(np.isnan(valid_outcomes), np.nan, valid_outcomes > 0)).sum(), len(valid_outcomes))
                print(pd.isna(np.where(np.isnan(valid_outcomes), np.nan, valid_outcomes > 0))[test_mask].sum(), len(valid_outcomes[test_mask]))
                test_slice = valid_df.encode_slice(sf.slices.SliceFeature("Phenylephrine", ["None"]).to_dict())
                test_mask = test_slice.make_mask(valid_df.df)
                print(pd.isna(np.where(np.isnan(valid_outcomes), np.nan, valid_outcomes > 0))[test_mask].sum(), len(valid_outcomes[test_mask]))
                model_metrics[f"{model_name} True"] = np.where(np.isnan(valid_outcomes), np.nan, valid_outcomes > 0)
                model_metrics[f"{model_name} Predicted"] = np.where(np.isnan(valid_preds), np.nan, valid_preds)
                self.metrics[model_name] = model_metrics
                
            metrics.update(self.metrics[model_name])
        
        rank_list = sf.slices.RankedSliceList(scored_slices, 
                                              valid_df, 
                                              score_fns, 
                                              similarity_threshold=0.7)
        rank_list.score_cache = self.eval_score_caches.setdefault(timestep_def, {})
        return rank_list, metrics, ids, valid_df
