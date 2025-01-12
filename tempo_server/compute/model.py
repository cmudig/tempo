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
from .raytuner import RayTuner
from .xgb import XGBoost
import torch
import logging

class Model:
    def __init__(self, model_fs, result_fs=None):
        super().__init__()
        self.fs = model_fs
        self.result_fs = result_fs if result_fs is not None else model_fs
        self.predictions = None
        self.metrics = None
        self.spec = None
        
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
        for fname in ["metrics.json", "model.json", "preds.pkl"]:
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
    
    def make_modeling_variables(self, query_engine, spec, update_fn=None, dummies=True):
        """Creates the variables dataframe."""
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
            logging.info(f"Before: {modeling_df.shape}")
            modeling_df = pd.get_dummies(modeling_df, 
                                        columns=[c for c in modeling_df.columns 
                                                if pd.api.types.is_object_dtype(modeling_df[c].dtype) 
                                                or pd.api.types.is_string_dtype(modeling_df[c].dtype) 
                                                or isinstance(modeling_df[c].dtype, pd.CategoricalDtype)])
            logging.info(f"After: {modeling_df.shape}")

        del modeling_variables
        return modeling_df

    def make_model(self, dataset, spec, modeling_df=None, update_fn=None):
        query_engine = dataset.make_query_engine()
        if modeling_df is None:
            if update_fn is not None: update_fn({'message': 'Loading variables'})
            modeling_df = self.make_modeling_variables(query_engine, spec, update_fn=update_fn)
            
        if update_fn is not None: update_fn({'message': 'Loading target variable'})
        outcome = query_engine.query("(" + spec['outcome'] + 
                                    (f" where ({spec['cohort']})" if spec.get('cohort', '') else '') + ") " + 
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
            outcome_values = codes
        else:
            outcome_values = outcome.get_values()
            
        train_ids, val_ids, test_ids = dataset.split_ids
        train_mask = outcome.get_ids().isin(train_ids)
        val_mask = outcome.get_ids().isin(val_ids)
        test_mask = outcome.get_ids().isin(test_ids)

        df_mask = pd.isna(modeling_df)
        # if df_mask.any().any():
        #     raise ValueError('There exist Nan values in the data')
        df_mask = ~df_mask
        df_mask = df_mask.all(axis=1)
        outcome_mask = ~pd.isna(outcome_values)
        row_mask = df_mask&outcome_mask
        
        # print(f'raw df mask {df_mask}')
        # print(f'df mask {df_mask}')
        # print(f'outcome mask {outcome_mask}')
        # print(f'row_mask {row_mask}')
        
        architecture_class, model, metrics, predictions = self._train_model(
            spec,
            modeling_df,
            outcome_values,
            outcome.get_ids(),
            train_mask,
            val_mask,
            test_mask,
            # row_mask=~pd.isna(outcome_values),
            row_mask=row_mask,
            update_fn=update_fn,
            model_type=spec["model_type"],
            early_stopping_rounds=3)
    
        # Save out the metadata
        self.fs.write_file(convert_to_native_types(spec),
                           "spec.json",
                           indent=2)
        
        # Save out the metrics    
        self.result_fs.write_file(convert_to_native_types(metrics),
                           "metrics.json")
            
        # Save out the model itself and its predictions

        if architecture_class == 'pytorch':
            dest_path = os.path.join(self.result_fs.base_path, 'model.pth')
            torch.save(model, dest_path)
            # with tempfile.NamedTemporaryFile('w+b', suffix='.json') as model_file:
            #     torch.save(model, model_file.name)
            #     model_file.seek(0)
        
        else:
            with tempfile.NamedTemporaryFile('r+', suffix='.json') as model_file:
                print(model_file.name)
                model.save_model(model_file.name)
                model_file.seek(0)
                self.result_fs.write_file(model_file.read(), "model.json")

        # Save out the true values and prediction (probabilities)
        self.result_fs.write_file(predictions, "preds.pkl")
        
        return model, metrics, predictions
    
    def _train_model(self, spec, variables, outcomes, ids, train_mask, val_mask, test_mask, model_type="binary_classification", row_mask=None, full_metrics=True, update_fn=None, **model_params):
        """
        variables: a dataframe containing variables for all patients
        """
        if row_mask is None: row_mask = np.ones(len(variables), dtype=bool)

        train_X = variables[train_mask & row_mask] #.values
        train_y = outcomes[train_mask & row_mask]
        train_ids = ids[train_mask & row_mask]
        logging.info(f"Training samples and missingness: {train_X}, {train_y}, {pd.isna(train_X).sum(axis=0)}, {pd.isna(train_y).sum()}")
        val_X = variables[val_mask & row_mask] #.values
        val_y = outcomes[val_mask & row_mask]
        val_ids = ids[val_mask & row_mask]
        logging.info(f"Val samples and missingness: {val_X}, {val_y}, {pd.isna(val_X).sum(axis=0)}, {pd.isna(val_y).sum()}")

        test_X = variables[test_mask & row_mask] #.values
        test_x = variables[test_mask & row_mask]
        test_y = outcomes[test_mask & row_mask]
        test_ids = ids[test_mask & row_mask]

        val_sample = np.random.uniform(size=len(val_X)) < 0.1
        
        model_architecture = spec.get("model_architecture", {}).get("type", "xgboost")
        if model_architecture != 'xgboost':
            config = spec.get("model_architecture", {}).get("hyperparameters", {})
            # config['num_epochs'] = {'type': 'fix', 'value': 10}
            config['input_size'] = {'type': 'fix', 'value': train_X.shape[1]}
            config['num_classes'] = {'type': 'fix', 'value': 1}
            model = NeuralNetwork(model_architecture,
                                  train_X,
                                  train_y,
                                  train_ids,
                                  test_x,
                                  test_y,
                                  test_ids,
                                  val_X,
                                  val_y,
                                  val_ids,
                                  config)
            architecture_class = 'pytorch'
            logging.info(f'model config {config}')
            logging.info("Training")
            model.train(tuning_mode = spec.get("model_architecture", {}).get("tuner", False))
        else: 
            architecture_class = 'xgboost'
            if model_type.endswith("classification"):
                train_y = train_y.astype(int)
                val_y = val_y.astype(int)
                test_y = test_y.astype(int)
            model = XGBoost(model_type, 
                            train_X,
                            train_y,
                            train_ids,
                            val_X,
                            val_y,
                            val_ids,
                            val_sample,
                            test_X,
                            test_y,
                            test_ids,
                            **model_params)
            logging.info("Training")
            model.train()
 
        logging.info("Evaluating")

        if update_fn is not None: update_fn({'message': 'Evaluating model'})
        metrics = model.evaluate(spec,full_metrics,variables,outcomes,train_mask,val_mask,test_mask,row_mask,**model_params)
        predict = model.predict()
        logging.info(f'predict type{type(predict)}, size {predict.shape}')
        logging.info(f'metrics {metrics}')
        
        # Return preds and true values in the validation and test sets, putting
        # nans whenever the row shouldn't be considered part of the
        # cohort for this model
        
        val_pred = model.predict(data_type='val')
        test_pred = model.predict()
        return architecture_class, model.model, metrics, (
            np.where(val_mask & row_mask, outcomes, np.nan)[val_mask],
            self._get_dataset_aligned_predictions(outcomes, val_pred, (val_mask & row_mask).values)[val_mask],
            np.where(test_mask & row_mask, outcomes, np.nan)[test_mask],
            self._get_dataset_aligned_predictions(outcomes, test_pred, (test_mask & row_mask).values)[test_mask],
        )
