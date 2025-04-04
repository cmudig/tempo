import pandas as pd
import re
import tempfile
import os
from tempo_server.query_language.data_types import *
from .utils import make_series_summary, make_query
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import r2_score, roc_auc_score, confusion_matrix, roc_curve, f1_score
from divisi.utils import convert_to_native_types
from .nn import NeuralNetwork
import shap
from .xgb import XGBoost
import uuid
import logging
import traceback

class Model:
    def __init__(self, model_fs, result_fs=None):
        super().__init__()
        self.fs = model_fs
        self.result_fs = result_fs if result_fs is not None else model_fs
        self.predictions = None
        self.metrics = None
        self.spec = None
        self._prediction_cache_fs = self.result_fs.subdirectory("predictions")
        self._prediction_cache = {'ids': {}, 'inputs': []}
        
    def get_spec(self):
        if self.spec is None:
            self.spec = self.fs.read_file("spec.json")
        return self.spec
    
    def write_draft_spec(self, draft):
        self.spec = None
        try:
            meta = self.get_spec()
        except:
            meta = draft
            
        if not len(draft):
            # Delete the draft
            if "draft" in meta: 
                del meta["draft"]
                self.fs.write_file(meta, "spec.json")
        else:
            self.fs.write_file({**meta, "draft": draft}, "spec.json")          
        
    def write_spec(self, new_spec):
        self.fs.write_file(new_spec, "spec.json")
        self.spec = new_spec
        
    def get_metrics(self):
        if self.metrics is None:
            self.metrics = self.result_fs.read_file("metrics.json")
        return self.metrics
    
    def get_true_labels(self, split):
        """split can be 'val' or 'test'."""
        assert split in ('val', 'test'), f"Unknown split for model labels: '{split}'"
        if not self.predictions:
            self.predictions = tuple(x.astype(np.float64) for x in self.result_fs.read_file("preds.pkl"))
        if split == 'val': return self.predictions[0]
        return self.predictions[2]
    
    def get_model_predictions(self, split):
        """split can be 'val' or 'test'."""
        assert split in ('val', 'test'), f"Unknown split for model labels: '{split}'"
        if not self.predictions:
            self.predictions = tuple(x.astype(np.float64) for x in self.result_fs.read_file("preds.pkl"))
        if split == 'val': return self.predictions[1]
        return self.predictions[3]
    
    def get_optimal_threshold(self):
        metrics = self.get_metrics()
        return metrics["threshold"]
    
    @property
    def is_trained(self):
        return "model_type" in self.get_spec() and self.result_fs.exists("metrics.json")
        
    @classmethod
    def blank_spec(cls, **kwargs):
        return {
            "variables": kwargs.get("variables", {"Untitled": {
                "category": "Inputs",
                "query": ""
            }}),
            "timestep_definition": kwargs.get("timestep_definition", ""),
            "cohort": kwargs.get("cohort", ""),
            "outcome": kwargs.get("outcome", ""),
            "description": kwargs.get("description", ""),
        }
        
    def copy_to(self, model_fs, result_fs=None):
        """Copies the contents to the given new locations, and returns a Model at the new locations."""
        if self.fs.exists("spec.json"):
            self.fs.copy_file(model_fs, "spec.json")
        result_fs = self.fs if result_fs is None else result_fs
        for fname in ["metrics.json", "model.json", "model.pth", "preds.pkl"]:
            if self.result_fs.exists(fname):
                self.result_fs.copy_file(result_fs, fname)
        return Model(model_fs, result_fs)
        
    def _compute_predictions(self, model, model_type, X):
        if model_type == "regression":
            return model.predict(X)
        elif model_type == "multiclass_classification":
            return model.predict_proba(X)
        else:
            return model.predict_proba(X)[:,1]

    def _get_dataset_aligned_predictions(self, all_outcomes, preds, apply_mask):
        result = np.empty((len(all_outcomes), preds.shape[1])) if len(preds.shape) > 1 else np.empty(len(all_outcomes))
        result.fill(np.nan)
        result[apply_mask] = preds
        return result
    
    def make_modeling_variables(self, query_engine, spec, update_fn=None, dummies=True, dummy_col_values=None):
        """
        Creates the variables dataframe.
        """
        query = make_query(spec["variables"], spec["timestep_definition"])
        logging.info(query)
        if update_fn is not None:
            def prog(num_completed, num_total):
                update_fn({'message': f'Loading variables ({num_completed} / {num_total})', 'progress': num_completed / num_total})
        else:
            prog = None
        modeling_variables = query_engine.query(query, update_fn=prog)
        modeling_df = modeling_variables.values

        if dummies:
            return self.convert_dummy_variables(modeling_df, dummy_col_values=dummy_col_values)
        
        return modeling_df
    
    def convert_dummy_variables(self, modeling_df, dummy_col_values=None):
        """
        If dummy_col_values is provided, it should be a dictionary of one-hot
        encoded column names to lists of values used for each column.
        """
        logging.info(f"Before: {modeling_df.shape}")
        if dummy_col_values is None:
            dummy_columns = [c for c in modeling_df.columns 
                            if pd.api.types.is_object_dtype(modeling_df[c].dtype) 
                            or pd.api.types.is_string_dtype(modeling_df[c].dtype) 
                            or isinstance(modeling_df[c].dtype, pd.CategoricalDtype)]
            modeling_df[dummy_columns] = modeling_df[dummy_columns].astype('category')
        else:
            dummy_columns = list(dummy_col_values.keys())
            for c, values in dummy_col_values.items():
                if (~pd.isna(modeling_df[c]) & ~modeling_df[c].isin(values)).any():
                    unknown_vals = modeling_df[c][~modeling_df[c].isin(values)]
                    raise ValueError(f"Unknown values {', '.join((str(x) for x in unknown_vals.unique()))} for feature {c}")
                modeling_df[c] = modeling_df[c].astype(pd.CategoricalDtype(values))
            
        new_modeling_df = pd.get_dummies(modeling_df, 
                                    columns=dummy_columns)
        if dummy_col_values is None:
            # keep track of which columns were added so we can create them later if needed
            dummy_col_values = {c: modeling_df[c].cat.categories.tolist()
                                for c in dummy_columns}
        modeling_df = new_modeling_df
        
        logging.info(f"After: {modeling_df.shape}")
        return modeling_df, dummy_col_values

    def make_model(self, dataset, spec, modeling_df=None, dummy_variables=None, update_fn=None):
        query_engine = dataset.make_query_engine()
        if modeling_df is None:
            if update_fn is not None: update_fn({'message': 'Loading variables'})
            modeling_df, dummy_variables = self.make_modeling_variables(query_engine, spec, update_fn=update_fn)
            
        if update_fn is not None: update_fn({'message': 'Loading target variable'})
        outcome = query_engine.query("((" + spec['outcome'] + 
                                    (f") where ({spec['cohort']})" if spec.get('cohort', '') else ')') + ") " + 
                                    spec["timestep_definition"])
        logging.info(f"Outcome missingness: {(~pd.isna(outcome.get_values())).sum()}")
        
        if update_fn is not None: update_fn({'message': 'Training model'})
        if "model_type" not in spec:
            num_unique = len(np.unique(outcome.get_values()[~pd.isna(outcome.get_values())]))
            if (pd.api.types.is_numeric_dtype(outcome.get_values().dtype) and 
                num_unique > 10):
                spec["model_type"] = "regression"
            elif (num_unique == 2 and 
                  pd.api.types.is_numeric_dtype(outcome.get_values().dtype) and 
                  set(np.unique(outcome.get_values()[~pd.isna(outcome.get_values())]).astype(int).tolist()) == set([0, 1])):
                spec["model_type"] = "binary_classification"
            else:
                spec["model_type"] = "multiclass_classification"
                
        if spec["model_type"] == "multiclass_classification":
            # Encode the outcome values as integers and save the mapping
            codes, uniques = pd.factorize(outcome.get_values(), sort=True)
            codes = np.where(codes >= 0, codes, np.nan)
            spec["output_values"] = list(uniques)
            logging.info(f"output values: {spec['output_values']}")
            outcome_values = codes
        else:
            outcome_values = outcome.get_values()
            
        train_ids, val_ids, test_ids = dataset.split_ids
        train_mask = outcome.get_ids().isin(train_ids)
        val_mask = outcome.get_ids().isin(val_ids)
        test_mask = outcome.get_ids().isin(test_ids)

        df_mask = pd.isna(modeling_df)
        df_mask = ~df_mask
        df_mask = df_mask.all(axis=1)
        outcome_mask = ~pd.isna(outcome_values)
        valid_mask = df_mask & outcome_mask
                
        model, metrics, predictions = self._train_model(
            spec,
            modeling_df,
            outcome_values,
            outcome.get_ids(),
            train_mask,
            val_mask,
            test_mask,
            valid_mask,
            update_fn=update_fn,
            early_stopping_rounds=3)
    
        # Save out the metadata
        self.fs.write_file(convert_to_native_types(spec),
                           "spec.json",
                           indent=2)
        
        # Save out the metrics    
        logging.info(f"Hyperparameters: {model.get_hyperparameters()}")
        self.result_fs.write_file({**convert_to_native_types(metrics), 
                                   'model_architecture': {
                                       "type": spec.get("model_architecture", {}).get("type", "xgboost"),
                                       "num_samples": spec.get("model_architecture", {}).get("num_samples", False),
                                       "hyperparameters": model.get_hyperparameters()
                                   },
                                   **({'dummy_variables': dummy_variables} if dummy_variables else {})},
                           "metrics.json")
            
        # Save out the model itself and its predictions

        model.save(self.result_fs)

        # Save out the true values and prediction (probabilities)
        self.result_fs.write_file(predictions, "preds.pkl")
        
        return model, metrics, predictions
    
    def _make_model_trainer(self, spec, input_size, output_size):
        model_architecture = spec.get("model_architecture", {}).get("type", "xgboost")
        config = spec.get("model_architecture", {}).get("hyperparameters", {})
        if model_architecture == 'rnn' or model_architecture == 'transformer' or model_architecture == 'dense':
            return NeuralNetwork(spec["model_type"], 
                                  model_architecture,
                                  config,
                                  input_size, 
                                  output_size)
        else: 
            return XGBoost(spec["model_type"], **config)

    def _train_model(self, spec, variables, outcomes, ids, train_mask, val_mask, test_mask, valid_mask, full_metrics=True, update_fn=None, **model_params):
        """
        variables: a dataframe containing variables for all patients
        valid_mask: denotes where neither the inputs nor output are missing
        """
        train_X = variables[train_mask & valid_mask] #.values
        train_y = outcomes[train_mask & valid_mask]
        train_ids = ids[train_mask & valid_mask]
        logging.info(f"Training samples and missingness: {train_X.shape}, {train_y.shape}, {pd.isna(train_X).sum(axis=0)}, {pd.isna(train_y).sum()}")
        val_X = variables[val_mask & valid_mask] #.values
        val_y = outcomes[val_mask & valid_mask]
        val_ids = ids[val_mask & valid_mask]
        logging.info(f"Val samples and missingness: {val_X.shape}, {val_y.shape}, {pd.isna(val_X).sum(axis=0)}, {pd.isna(val_y).sum()}")

        test_X = variables[test_mask & valid_mask] #.values
        test_y = outcomes[test_mask & valid_mask]
        test_ids = ids[test_mask & valid_mask]
        logging.info(f"Test samples and missingness: {test_X.shape}, {test_y.shape}, {pd.isna(test_X).sum(axis=0)}, {pd.isna(test_y).sum()}")
        
        if spec["model_type"].endswith("classification"):
            train_y = train_y.astype(int)
            val_y = val_y.astype(int)
            test_y = test_y.astype(int)

        model = self._make_model_trainer(spec, train_X.shape[1], train_y.max() + 1 if spec["model_type"] == 'multiclass_classification' else 1)
        logging.info("Training")
        try:
            model.train(
                train_X,
                train_y,
                train_ids,
                val_X,
                val_y,
                val_ids,
                progress_fn=lambda info: update_fn({'message': f"Training ({info.get('message', '...')})"}),
                num_samples=spec.get("model_architecture", {}).get("num_samples", 1)
            )
        except Exception as e:
            logging.info(f"Model training error: {traceback.format_exc()}")
            raise ValueError(f"Error training model: {e}")
       
        logging.info("Evaluating")

        if update_fn is not None: update_fn({'message': 'Evaluating model'})
        metrics = model.evaluate(spec,
                                 full_metrics,
                                 variables,
                                 outcomes,
                                 ids,
                                 train_mask & valid_mask,
                                 val_mask & valid_mask,
                                 test_mask & valid_mask,
                                 progress_fn=update_fn)
        logging.info(f'metrics {metrics}')
        
        val_pred = model.predict(val_X, val_ids)
        test_pred = model.predict(test_X, test_ids)
        logging.info(f'predict type{type(test_pred)}, size {test_pred.shape}')

        # Return preds and true values in the validation and test sets, putting
        # nans whenever the row shouldn't be considered part of the
        # cohort for this model
        return model, metrics, (
            np.where(val_mask, outcomes, np.nan)[val_mask],
            self._get_dataset_aligned_predictions(outcomes, val_pred, (val_mask & valid_mask).values)[val_mask],
            np.where(test_mask, outcomes, np.nan)[test_mask],
            self._get_dataset_aligned_predictions(outcomes, test_pred, (test_mask & valid_mask).values)[test_mask],
        )

    def load_cache_if_needed(self):
        try:    
            self._prediction_cache = self._prediction_cache_fs.read_file("cache.json")
        except:
            self._prediction_cache = {'ids': {}, 'inputs': []}
                
    def get_modeling_inputs(self, dataset, ids, update_fn=None):
        spec = self.get_spec()
        metrics = self.get_metrics()
        if "dummy_variables" not in metrics:
            raise ValueError("Model was not trained with dummy variables saved, please re-train")
        dv = metrics["dummy_variables"]
        
        # First get modeling variables, then filter down to requested IDs
        query_engine = dataset.make_query_engine()
        if update_fn is not None: update_fn({'message': 'Loading variables'})
        modeling_df, _ = self.make_modeling_variables(query_engine, spec, update_fn=update_fn, dummy_col_values=dv)
            
        if update_fn is not None: update_fn({'message': 'Loading target variable'})
        outcome = query_engine.query("((" + spec['outcome'] + 
                                    (f") where ({spec['cohort']})" if spec.get('cohort', '') else ')') + ") " + 
                                    spec["timestep_definition"])
        matching_ids = outcome.get_ids().isin(dataset.get_numerical_ids(ids))
        modeling_df = modeling_df[matching_ids].reset_index(drop=True)
        outcome = outcome.filter(matching_ids)
        if set(outcome.get_ids().tolist()) != set(dataset.get_numerical_ids(ids)):
            missing_ids = dataset.get_original_ids(np.array(list(set(dataset.get_numerical_ids(ids)) - set(outcome.get_ids().tolist()))))
            raise ValueError(f"Some IDs are not present in dataset: {', '.join(str(s) for s in missing_ids)}")
        
        # if outcome is present, exclude rows where outcome is NA
        modeling_df = modeling_df[~pd.isna(outcome.get_values())]
        return_ids = pd.DataFrame({'id': outcome.get_ids(), 'time': outcome.get_times()}).to_dict(orient='records')
        outcome = outcome.get_values()
        logging.info(f"{modeling_df.shape}, {outcome.shape}")
            
        return modeling_df, outcome, return_ids

    def lookup_prediction_results(self, ids=None, inputs=None, n_feature_importances=5):
        """
        Returns the results of the model prediction operation (as a list of
        records) if they have already been computed. If the operation resulted in an error,
        raises that error as a ValueError."""
        self.load_cache_if_needed()
        
        if (ids is None) == (inputs is None):
            raise ValueError("Exactly one of ids or inputs must be provided")
        
        if ids is not None:
            result_key = 'ids'
            result_id = ",".join(str(x) for x in ids)
            result_idx, matching_result = next(((i, result) for i, result in enumerate(self._prediction_cache['ids'].get(result_id, []))
                                    if result['n_feature_importances'] == n_feature_importances), (None, None))
        elif inputs is not None:
            result_key = 'inputs'
            result_id = None
            result_idx, matching_result = next(((i, result) for i, result in enumerate(self._prediction_cache['inputs'])
                                                if result['inputs'] == inputs and result['n_feature_importances'] == n_feature_importances), (None, None))
        
        if matching_result:
            try:
                results_json = self._prediction_cache_fs.read_file(matching_result["path"])
            except:
                logging.info("Prediction results file was removed")
                # file was removed
                if result_id is not None:
                    del self._prediction_cache[result_key][result_id][result_idx]
                else:
                    del self._prediction_cache[result_key][result_idx]
                self.write_cache()
            else:
                if isinstance(results_json, dict) and "error" in results_json:
                    raise ValueError(results_json["error"])
                
                return results_json
        return None

    def compute_model_predictions(self, dataset, ids=None, inputs=None, update_fn=None, n_feature_importances=5):
        """
        Compute model predictions for instances corresponding to the given 
        trajectory IDs or inputs.
        """
        if (ids is None) == (inputs is None):
            raise ValueError("Exactly one of ids or inputs must be provided")
        
        try:
            results = self.lookup_prediction_results(ids=ids, inputs=inputs)
            if results:
                return results
        except Exception as e:
            logging.info(f"Error loading cached prediction results: {e}")
            
        spec = self.get_spec()
        metrics = self.get_metrics()
        
        query_engine = dataset.make_query_engine()
        background_df, dv = self.make_modeling_variables(query_engine, 
                                                         spec, update_fn=update_fn)
        background_outcome = query_engine.query("((" + spec['outcome'] + 
                                    (f") where ({spec['cohort']})" if spec.get('cohort', '') else ')') + ") " + 
                                    spec["timestep_definition"])
        
        if ids is not None:
            modeling_df, outcome, return_ids = self.get_modeling_inputs(dataset, ids, update_fn=update_fn)
            modeling_ids = np.array([x["id"] for x in return_ids])
            logging.info(f"{modeling_df.shape}, {outcome.shape}")
            
        elif inputs is not None:
            # use the order of the modeling spec variables
            if update_fn is not None: update_fn({'message': "Preprocessing data"})
            modeling_df = pd.DataFrame.from_records(inputs)
            if "id" in inputs.columns:
                # remove the special ID column
                modeling_ids = modeling_df["id"]
                modeling_df = modeling_df.drop(columns=["id"])
            else:
                modeling_ids = np.arange(len(modeling_df))
            if any(v not in modeling_df.columns for v in spec["variables"].keys()):
                raise ValueError(f"Required input features {', '.join(v for v in spec['variables'] if v not in modeling_df.columns)} not present in inputs")
            modeling_df, _ = self.convert_dummy_variables(modeling_df, dummy_col_values=dv)
            
            outcome = None
            return_ids = inputs

        # load the model
        if update_fn is not None: update_fn({'message': "Loading model"})
        model = self._make_model_trainer(spec, 
                                         modeling_df.shape[1],
                                         len(spec["output_values"]) if "output_values" in spec else 1)
        model.load(metrics["model_architecture"]["hyperparameters"], self.result_fs)
        logging.info("Loaded model")
        
        if update_fn is not None: update_fn({'message': "Running model"})
        preds = model.predict(modeling_df, modeling_ids).tolist()
        
        if n_feature_importances > 0:
            try:
                shap_values = model.explain(modeling_df, modeling_ids, background_df, background_outcome.get_ids())
            except Exception as e:
                logging.info(f"Error getting SHAP values for prediction: {traceback.format_exc()}")
                shap_values = None
        else:
            shap_values = None
        
        results = convert_to_native_types([
            {'input': return_ids[i],
             'prediction': preds[i],
             **({'ground_truth': outcome[i]} if outcome is not None else {}),
             **({'feature_importances': [
                 {'feature': modeling_df.columns[f], 'importance': shap_values[i, f]}
                 for f in np.flip(np.argsort(np.abs(shap_values[i])))[:n_feature_importances]
             ]} if shap_values is not None else {})
             }
            for i in range(len(return_ids))
        ])
        
        path = uuid.uuid4().hex + ".json"
        self._prediction_cache_fs.write_file(results, path)
        self.load_cache_if_needed()
        if ids is not None:
            self._prediction_cache['ids'].setdefault(",".join(str(x) for x in ids), []).append({
                "n_feature_importances": n_feature_importances,
                "path": path
            })
        elif inputs is not None:
            self._prediction_cache['inputs'].append({
                "inputs": inputs,
                "n_feature_importances": n_feature_importances,
                "path": path
            })
        self.write_cache()
        
        return results
        
    def write_cache(self):
        self._prediction_cache_fs.write_file(self._prediction_cache, "cache.json")
        
