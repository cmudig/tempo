import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import r2_score, roc_auc_score, confusion_matrix, roc_curve, f1_score
from tempo_server.query_language.data_types import *
from .utils import make_series_summary
import shap
import traceback
import logging
import tempfile

class XGBoost:
    def __init__(self,model_type,**model_params):
        self.model_cls = xgb.XGBRegressor if model_type == "regression" else xgb.XGBClassifier
        # Don't do class weights - instead, we can simply choose a better operating point
        # if not regressor:
        #     model_params['scale_pos_weight'] = (len(train_y) - train_y.sum()) / train_y.sum()
        
        self.model_type = model_type
        
        self.model = self.model_cls(**model_params)
        self.model_params = self.model.get_xgb_params()
    
    def load(self, hyperparameter_info, model_fs):
        with tempfile.NamedTemporaryFile('r+', suffix='.json') as model_file:
            contents = model_fs.read_file(model_file.read(), "model.json")
            model_file.write(contents)
            model_file.seek(0)
            self.model.load_model(model_file.name)
            
    def save(self, model_fs):
        with tempfile.NamedTemporaryFile('r+', suffix='.json') as model_file:
            print(model_file.name)
            self.model.save_model(model_file.name)
            model_file.seek(0)
            model_fs.write_file(model_file.read(), "model.json")
     
    def get_hyperparameters(self):
        """Create a dictionary of hyperparameters that can be used to reinstantiate the model."""
        return self.model_params
               
    def train(self, train_X, train_y, train_ids, val_X, val_y, val_ids, progress_fn=None, use_tuner=False):
        if self.model_type == "multiclass_classification":
            weights = compute_sample_weight(
                class_weight='balanced',
                y=train_y
            )
            params = {'sample_weight': weights}
        else:
            params = {}
        
        self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)], **params)
        self.model_params = self.model.get_xgb_params()

    def predict(self, test_X, test_ids):
        if self.model_type == "regression":
            return self.model.predict(test_X)
        elif self.model_type == "multiclass_classification":
            return self.model.predict_proba(test_X)
        else:
            return self.model.predict_proba(test_X)[:,1]
    
    def explain(self, test_X, test_ids):
        """Returns the SHAP values for the given set of instances as a matrix,
        one row per input and one column per feature."""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(test_X)
        return shap_values
        
    def evaluate(self, spec, full_metrics, variables, outcomes, ids, train_mask, val_mask, test_mask):
        test_X = variables[test_mask]
        test_pred = self.predict(test_X, ids[test_mask])
        test_y = outcomes[test_mask]
        metrics = {}
        metrics["labels"] = make_series_summary(test_y 
                                                     if self.model_type != 'multiclass_classification' 
                                                     else pd.Series([spec["output_values"][i] for i in test_y]))
        if self.model_type == "regression":
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
                if self.model_type == "binary_classification":
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
                elif self.model_type == "multiclass_classification":
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
                    } for i, c in enumerate(spec["output_values"])]
                    metrics["predictions"] = make_series_summary(pd.Series([spec["output_values"][i] for i in max_predictions]))

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
                
        metrics["n_train"] = {"instances": train_mask.sum(), "trajectories": len(np.unique(ids[train_mask]))}
        metrics["n_val"] = {"instances": val_mask.sum(), "trajectories": len(np.unique(ids[val_mask]))}
        metrics["n_test"] = {"instances": test_mask.sum(), "trajectories": len(np.unique(ids[test_mask]))}

        try:
            shap_values = self.explain(test_X, ids[test_mask])
            perf = np.mean(shap_values,axis=0)
            perf_std = np.std(shap_values,axis=0)
            sorted_perf_index = np.flip(np.argsort(perf))
            metrics['feature_importances'] = [{'feature': variables.columns[i], 'mean': perf[i], 'std': perf_std[i]} 
                                             for i in sorted_perf_index]
        except:
            logging.error(traceback.format_exc())
            print("Error calculating shap values")
                    
        if submodel_metric is not None and full_metrics:
            # Check for trivial solutions
            max_variables = 5
            metric_fraction = 0.95
            
            variable_names = []
            for i in reversed(np.argsort(self.model.feature_importances_)[-max_variables:]):
                variable_names.append(variables.columns[i])
                smaller_model = XGBoost(self.model_type, **self.model_params)
                smaller_model.train(variables[train_mask],
                                        outcomes[train_mask],
                                        ids[train_mask],
                                        variables[val_mask],
                                        outcomes[val_mask],
                                        ids[val_mask])
                sub_metrics = smaller_model.evaluate(spec,False,variables,outcomes,ids,train_mask,val_mask,test_mask)
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
        return metrics
    