import pandas as pd
import os
import json
import re
import pickle
from query_language.data_types import *
from utils import make_series_summary, make_query
import xgboost
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import r2_score, roc_auc_score, confusion_matrix, roc_curve, f1_score
import slice_finding as sf

def make_modeling_variables(dataset, variable_definitions, timestep_definition):
    """Creates the variables dataframe."""
    query = make_query(variable_definitions, timestep_definition)
    print(query)
    modeling_variables = dataset.query(query)
    modeling_df = modeling_variables.values

    print("Before:", modeling_df.shape)
    print([c for c in modeling_df.columns 
                                          if pd.api.types.is_object_dtype(modeling_df[c].dtype) 
                                          or pd.api.types.is_string_dtype(modeling_df[c].dtype) 
                                          or isinstance(modeling_df[c].dtype, pd.CategoricalDtype)])
    print(modeling_df.dtypes)
    modeling_df = pd.get_dummies(modeling_df, 
                                 columns=[c for c in modeling_df.columns 
                                          if pd.api.types.is_object_dtype(modeling_df[c].dtype) 
                                          or pd.api.types.is_string_dtype(modeling_df[c].dtype) 
                                          or isinstance(modeling_df[c].dtype, pd.CategoricalDtype)])
    print("After:", modeling_df.shape)

    del modeling_variables
    return modeling_df

def make_data_summary_field(field, model_meta, dataset, row_mask):
    if field.get("type", None) == "group":
        subresults = [make_data_summary_field(c, model_meta, dataset, row_mask) for c in field["children"]]
        if "sort" in field:
            subresults = sorted(subresults, key=lambda x: x["summary"][field["sort"]], reverse=not field.get("ascending", True))
        if "topk" in field:
            subresults = subresults[:field["topk"]]
        return {"name": field.get("name", ""), "type": "group", "children": subresults}
    if "query" in field:
        query = ("(" + field['query'] + ") " +
                #  " " + (f"where ({model_meta['cohort']})" if model_meta.get('cohort', '') else '') + ") " + 
                 model_meta["timestep_definition"])
        print(query)
        query_result = dataset.query(query) 
        if not field.get("per_timestep", False):
            query_result = pd.DataFrame({"id": query_result.get_ids()[row_mask], 
                                         "value": query_result.get_values()[row_mask]}).drop_duplicates("id")["value"]
        else:
            query_result = query_result.get_values()[row_mask]
        return {"name": field.get("name", field["query"]), 
                "type": "variable_summary", 
                "summary": make_series_summary(query_result)}
        
    raise ValueError(f"Unknown field type {field}")
    
def make_model_data_summary(data_summary_config, model_meta, dataset, row_mask):
    """Creates a data summary for the rows in the dataset that match the model's
    inclusion criteria."""
    summary = {
        "fields": []
    }
    if (fields := data_summary_config.get("fields", None)) is not None:
        for field in fields:
            summary["fields"].append(make_data_summary_field(field, model_meta, dataset, row_mask))
    return summary
    
class ModelTrainer:
    def __init__(self, config, data_dir, model_dir, slices_dir):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.slices_dir = slices_dir
        
    def _get_predictions(self, model, model_type, X):
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
        
    def _train_model(self, model_meta, variables, outcomes, ids, train_mask, val_mask, test_mask, model_type="binary_classification", columns_to_drop=None, columns_to_add=None, row_mask=None, full_metrics=True, **model_params):
        """
        variables: a dataframe containing variables for all patients
        """
        variables = variables.drop(columns=[c for c in variables.columns
                                            if columns_to_drop is not None and re.search(columns_to_drop, c) is not None])
        if row_mask is None: row_mask = np.ones(len(variables), dtype=bool)
        train_X = variables[train_mask & row_mask].values
        train_y = outcomes[train_mask & row_mask]
        print(train_X, train_y, pd.isna(train_X).sum(axis=0), pd.isna(train_y).sum())
        train_ids = ids[train_mask & row_mask]
        val_X = variables[val_mask & row_mask].values
        val_y = outcomes[val_mask & row_mask]
        print(val_X, val_y, pd.isna(val_X).sum(axis=0), pd.isna(val_y).sum())
        val_ids = ids[val_mask & row_mask]
        test_X = variables[test_mask & row_mask].values
        test_y = outcomes[test_mask & row_mask]
        test_ids = ids[test_mask & row_mask]
        if model_type.endswith("classification"):
            train_y = train_y.astype(int)
            val_y = val_y.astype(int)
            test_y = test_y.astype(int)
        if columns_to_add is not None:
            train_X = np.hstack([train_X, columns_to_add[train_mask & row_mask].values])
            val_X = np.hstack([val_X, columns_to_add[val_mask & row_mask].values])
            test_X = np.hstack([test_X, columns_to_add[test_mask & row_mask].values])
        val_sample = np.random.uniform(size=len(val_X)) < 0.1
        
        print("Training", train_X.shape)
        model_cls = xgboost.XGBRegressor if model_type == "regression" else xgboost.XGBClassifier
        # Don't do class weights - instead, we can simply choose a better operating point
        # if not regressor:
        #     model_params['scale_pos_weight'] = (len(train_y) - train_y.sum()) / train_y.sum()
        if model_type == "multiclass_classification":
            weights = compute_sample_weight(
                class_weight='balanced',
                y=train_y
            )
            params = {'sample_weight': weights}
        else:
            params = {}
            
        model = model_cls(**model_params)
        model.fit(train_X, train_y, eval_set=[(val_X[val_sample], val_y[val_sample])], **params)
        
        print("Evaluating")
            
        test_pred = self._get_predictions(model, model_type, test_X)
        metrics = {}
        metrics["labels"] = make_series_summary(test_y 
                                                     if model_type != 'multiclass_classification' 
                                                     else pd.Series([model_meta["output_values"][i] for i in test_y]))
        if model_type == "regression":
            metrics["performance"] = {
                "R^2": float(r2_score(test_y, test_pred)),
                "MSE": float(np.mean((test_y - test_pred) ** 2))
            }
            bin_edges = np.histogram_bin_edges(np.concatenate([test_y, test_pred]), bins=10)
            metrics["hist"] = {
                "values": np.histogram2d(test_y, test_pred, bins=bin_edges)[0].tolist(),
                "bins": bin_edges.tolist()
            }
            hist, bin_edges = np.histogram((test_pred - test_y), bins=10)
            metrics["difference_hist"] = {
                "values": hist.tolist(),
                "bins": bin_edges.tolist()
            }
            metrics["predictions"] = make_series_summary(test_pred)

            submodel_metric = "R^2"
        else:
            test_y = test_y.astype(np.uint8)
            if len(np.unique(test_y)) > 1:
                if model_type == "binary_classification":
                    fpr, tpr, thresholds = roc_curve(test_y, test_pred)
                    opt_threshold = thresholds[np.argmax(tpr - fpr)]
                    if np.isinf(opt_threshold):
                        # Set to 1 if positive label is never predicted, otherwise 0
                        if (test_y > 0).mean() < 0.01:
                            opt_threshold = 1e-6
                        else:
                            opt_threshold = 1 - 1e-6
                        
                    metrics["threshold"] = float(opt_threshold)
                    metrics["performance"] = {
                        "Accuracy": float((test_y == (test_pred >= opt_threshold)).mean()),
                        "AUROC": float(roc_auc_score(test_y, test_pred)),
                        "Micro F1": float(f1_score(test_y, test_pred >= opt_threshold, average="micro")),
                        "Macro F1": float(f1_score(test_y, test_pred >= opt_threshold, average="macro")),
                    }
                    metrics["roc"] = {}
                    for t in sorted([*np.arange(0, 1, 0.02), 
                                    float(opt_threshold)]):
                        conf = confusion_matrix(test_y, (test_pred >= t))
                        tn, fp, fn, tp = conf.ravel()
                        metrics["roc"].setdefault("thresholds", []).append(round(t, 3))
                        metrics["roc"].setdefault("fpr", []).append(fp / (fp + tn))
                        metrics["roc"].setdefault("tpr", []).append(tp / (tp + fn))
                        precision = float(tp / (tp + fp))
                        recall = float(tp / (tp + fn))
                        metrics["roc"].setdefault("performance", []).append({
                            "Accuracy": (tp + tn) / conf.sum(),
                            "Sensitivity": recall,
                            "Specificity": float(tn / (tn + fp)),
                            "Precision": precision,
                            "Micro F1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0,
                            "Macro F1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0,
                        })
                    
                    conf = confusion_matrix(test_y, (test_pred >= opt_threshold))
                    metrics["confusion_matrix"] = conf.tolist()
                    tn, fp, fn, tp = conf.ravel()
                    metrics["performance"]["Sensitivity"] = float(tp / (tp + fn))
                    metrics["performance"]["Specificity"] = float(tn / (tn + fp))
                    metrics["performance"]["Precision"] = float(tp / (tp + fp))
                    metrics["predictions"] = make_series_summary(test_pred)

                    submodel_metric = "AUROC"
                elif model_type == "multiclass_classification":
                    max_predictions = np.argmax(test_pred, axis=1)
                    metrics["performance"] = {
                        "Accuracy": (test_y == max_predictions).mean(),
                        "Micro F1": float(f1_score(test_y, max_predictions, average="micro")),
                        "Macro F1": float(f1_score(test_y, max_predictions, average="macro")),
                    }
                    conf = confusion_matrix(test_y, max_predictions)
                    metrics["confusion_matrix"] = conf.tolist()
                    
                    metrics["perclass"] = [{
                        "label": c,
                        "performance": {
                            "Sensitivity": float(((max_predictions == i) & (test_y == i)).sum() / 
                                                (((max_predictions == i) & (test_y == i)).sum() + 
                                                ((max_predictions != i) & (test_y == i)).sum())),
                            "Specificity": float(((max_predictions != i) & (test_y != i)).sum() / 
                                                (((max_predictions != i) & (test_y != i)).sum() + 
                                                ((max_predictions == i) & (test_y != i)).sum())),
                            "Precision": float(((max_predictions == i) & (test_y == i)).sum() / 
                                                (((max_predictions == i) & (test_y == i)).sum() + 
                                                ((max_predictions == i) & (test_y != i)).sum()))
                        }
                    } for i, c in enumerate(model_meta["output_values"])]
                    metrics["predictions"] = make_series_summary(pd.Series([model_meta["output_values"][i] for i in max_predictions]))

                    submodel_metric = "Macro F1"
                
                if full_metrics:
                    # Check whether any classes are never predicted
                    class_true_positive_threshold = 0.1
                    for true_class, probs in enumerate(conf):
                        tp_fraction = probs[true_class] / probs.sum()
                        if tp_fraction < class_true_positive_threshold:
                            metrics.setdefault("class_not_predicted_warnings", []).append({
                                "class": true_class,
                                "true_positive_fraction": tp_fraction,
                                "true_positive_threshold": class_true_positive_threshold
                            })
            else:
                submodel_metric = None
                
        metrics["n_train"] = {"instances": len(train_X), "trajectories": len(np.unique(train_ids))}
        metrics["n_val"] = {"instances": len(val_X), "trajectories": len(np.unique(val_ids))}
        metrics["n_test"] = {"instances": len(test_X), "trajectories": len(np.unique(test_ids))}
        
        if submodel_metric is not None and full_metrics:
            # Check for trivial solutions
            max_variables = 5
            metric_fraction = 0.95
            
            variable_names = []
            for i in reversed(np.argsort(model.feature_importances_)[-max_variables:]):
                variable_names.append(variables.columns[i])
                _, sub_metrics, _ = self._train_model(
                    model_meta,
                    variables[variable_names],
                    outcomes,
                    ids,
                    train_mask,
                    val_mask,
                    test_mask,
                    row_mask=row_mask, 
                    model_type=model_type,
                    full_metrics=False,
                    **model_params)
                if sub_metrics["performance"][submodel_metric] >= metrics["performance"][submodel_metric] * metric_fraction:
                    if len(variable_names) <= len(variables.columns) * 0.5:
                        metrics["trivial_solution_warning"] = {
                            "metric": submodel_metric,
                            "variables": variable_names,
                            "metric_value": sub_metrics["performance"][submodel_metric],
                            "metric_threshold": metrics["performance"][submodel_metric] * metric_fraction,
                            "metric_fraction": metric_fraction
                        }
                    break
        
        # Return preds and true values in the validation and test sets, putting
        # nans whenever the row shouldn't be considered part of the
        # cohort for this model
        val_pred = self._get_predictions(model, model_type, val_X)
        return model, metrics, (
            np.where(val_mask & row_mask, outcomes, np.nan)[val_mask],
            self._get_dataset_aligned_predictions(outcomes, val_pred, (val_mask & row_mask).values)[val_mask],
            np.where(test_mask & row_mask, outcomes, np.nan)[test_mask],
            self._get_dataset_aligned_predictions(outcomes, test_pred, (test_mask & row_mask).values)[test_mask],
        )

    def make_model(self, dataset, model_meta, train_ids, val_ids, test_ids, modeling_df=None, save_name=None):
        if modeling_df is None:
            modeling_df = make_modeling_variables(dataset, model_meta["variables"], model_meta["timestep_definition"])
            
        outcome = dataset.query("(" + model_meta['outcome'] + 
                                    (f" where ({model_meta['cohort']})" if model_meta.get('cohort', '') else '') + ") " + 
                                    model_meta["timestep_definition"])
        print((~pd.isna(outcome.get_values())).sum())
        
        if "model_type" not in model_meta:
            num_unique = len(np.unique(outcome.get_values()[~pd.isna(outcome.get_values())]))
            if (pd.api.types.is_numeric_dtype(outcome.get_values().dtype) and 
                num_unique > 10):
                model_meta["model_type"] = "regression"
            elif (num_unique == 2 and 
                  pd.api.types.is_numeric_dtype(outcome.get_values().dtype) and 
                  set(np.unique(outcome.get_values()[~pd.isna(outcome.get_values())]).astype(int).tolist()) == set([0, 1])):
                model_meta["model_type"] = "binary_classification"
            else:
                model_meta["model_type"] = "multiclass_classification"
                
        if model_meta["model_type"] == "multiclass_classification":
            # Encode the outcome values as integers and save the mapping
            codes, uniques = pd.factorize(outcome.get_values(), sort=True)
            codes = np.where(codes >= 0, codes, np.nan)
            model_meta["output_values"] = list(uniques)
            outcome_values = codes
        else:
            outcome_values = outcome.get_values()
            
        train_mask = outcome.get_ids().isin(train_ids)
        val_mask = outcome.get_ids().isin(val_ids)
        test_mask = outcome.get_ids().isin(test_ids)
        
        model, metrics, predictions = self._train_model(
            model_meta,
            modeling_df,
            outcome_values,
            outcome.get_ids(),
            train_mask,
            val_mask,
            test_mask,
            row_mask=~pd.isna(outcome_values),
            model_type=model_meta["model_type"],
            early_stopping_rounds=3)
        
        if "models" in self.config and "data_summary" in self.config["models"]:
            metrics["data_summary"] = make_model_data_summary(self.config["models"]["data_summary"], model_meta, dataset, train_mask & ~pd.isna(outcome.get_values()))
            
        if save_name is not None:
            # Save out the metadata
            with open(os.path.join(self.model_dir, f"spec_{save_name}.json"), "w") as file:
                json.dump(sf.utils.convert_to_native_types(model_meta), file, indent=2)
            
            # Save out the metrics    
            with open(os.path.join(self.model_dir, f"metrics_{save_name}.json"), "w") as file:
                json.dump(sf.utils.convert_to_native_types(metrics), file, indent=2)
                
            # Save out the model itself and its predictions
            model.save_model(os.path.join(self.model_dir, f"model_{save_name}.json"))
            
            # Save out the true values and prediction (probabilities)
            with open(os.path.join(self.model_dir, f"preds_{save_name}.pkl"), "wb") as file:
                pickle.dump(predictions, file)
            
        return model, metrics, predictions
