import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch #pytorch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import r2_score, roc_auc_score, confusion_matrix, roc_curve, f1_score
import os
from scipy.stats import percentileofscore
import tempfile
import ray
from ray import train, tune
from ray.tune import Tuner
from ray.train import Checkpoint
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
import logging
import traceback

import shap
from .pytorchmodels import LSTM,TimeSeriesTransformer, DenseModel

from sklearn.metrics import f1_score, roc_curve, roc_auc_score, auc, precision_score, recall_score, precision_recall_curve, confusion_matrix, average_precision_score

from .utils import make_series_summary

class TrajectoryDataset(Dataset):
    def __init__(self,data, labels, ids):
        self.data = data
        self.labels = labels
        self.ids = ids
    
        self.id_pos = []
        last_id = None
        for i, id in enumerate(self.ids):
            if last_id != id:
                if self.id_pos:
                    self.id_pos[-1] = (self.id_pos[-1][0], i)
                    assert i > self.id_pos[-1][0], last_id
                self.id_pos.append((i, 0))
                last_id = id
        self.id_pos[-1] = (self.id_pos[-1][0], len(self.ids))
   
    def __len__(self):
        return len(self.id_pos)
                
    def __getitem__(self, idx):
        trajectory_indexes = np.arange(*self.id_pos[idx])
        assert len(trajectory_indexes) > 0
        
        sample = self.data[trajectory_indexes]
        label = self.labels[trajectory_indexes]
        
        return torch.tensor(sample,dtype=torch.float32),torch.tensor(label,dtype=torch.float32)

class DataNormalizer:
    def __init__(self, normalization_spec=None):
        self.normalization_spec = normalization_spec
        
    def fit(self, data):
        """
        Produce a normalization spec for each feature in the data, which is 
        an array of dictionaries, each containing a field 'variable' for the name
        of the column, 'method' ('binary', 'log' or 'linear'), and optional
        parameters for the method.
        """
        spec = []
        for col in data.columns:
            values = data[col]
            if not pd.api.types.is_numeric_dtype(values.dtype):
                raise ValueError(f"Inputs to ML model must be numeric")
            values = values.fillna(0)
            num_unique = len(np.unique(values))
            try:
                is_binary = num_unique == 2 and set(np.unique(values).astype(int).tolist()) == set([0, 1])
            except:
                is_binary = False
            if is_binary:
                spec.append({
                    "variable": col,
                    "method": "binary"
                })
            else:
                mean_percentile = percentileofscore(values, values.mean())
                if mean_percentile < 20 and (values >= 0).all():
                    # log scale - if zeros are present, convert them to randomly
                    # sampled values in the lower 5 percentile of the distribution
                    upper_bound = np.quantile(values, 0.05)
                    lower_bound = upper_bound / 1e3
                    scaled_scores = np.log(np.where(values == 0, np.random.uniform(lower_bound, upper_bound, size=values.shape), values))
                    spec.append({
                        "variable": col,
                        "method": "log",
                        "zero_lb": lower_bound,
                        "zero_ub": upper_bound,
                        "mean": scaled_scores.mean(),
                        "std": scaled_scores.std()
                    })
                else:
                    spec.append({
                        "variable": col,
                        "method": "linear",
                        "mean": values.mean(),
                        "std": values.std()
                    })
        self.normalization_spec = spec
        return spec        
        
    def transform(self, data):
        """Transforms the given data using this object's normalization spec, and
        returns a numpy array."""
        new_data = []
        for spec_element in self.normalization_spec:
            if spec_element["variable"] not in data.columns:
                raise ValueError(f"Variable {spec_element['variable']} missing from input data")
            values = data[spec_element["variable"]].values
            if spec_element["method"] == "binary":
                new_data.append(values - 0.5)
            elif spec_element["method"] == "linear":
                new_data.append((values - spec_element["mean"]) / spec_element["std"])
            elif spec_element["method"] == "log":
                scaled_scores = np.log(np.where(values == 0, np.random.uniform(spec_element["zero_lb"], spec_element["zero_ub"], size=values.shape), values))
                new_data.append((scaled_scores - spec_element["mean"]) / spec_element["std"])
        new_data = np.vstack(new_data).T
        return np.where(np.logical_or(np.isnan(new_data), np.isinf(new_data)), 0, new_data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
            
class NeuralNetwork:
    """
    Manages the training of a neural network classifier/regressor using pytorch.
    """
    def __init__(self,model_type, architecture, config, input_size, num_classes=1, tune_reporting=False):

        tmp_config = {}
        self.default_config = {
            'num_epochs': 10,
            'batch_size': 64,
            'lr': 0.001,
            'num_layers': 1,
            'num_heads' : 4,
            'hidden_dim': 128,
            'dropout': 0.1
        }

        self.config = {k: config.get(k, { "type": "fix", "value": self.default_config[k] })
                       for k in self.default_config}
        self.tune_reporting = tune_reporting

        self.best_config = None # populated when the model is trained or loaded

        self.model_type = model_type
        self.architecture = architecture
        self.num_classes = num_classes
        self.input_size = input_size

        self.model = None # only populated when the model is trained or loaded
        self.data_normalizer = DataNormalizer() # populated when the model is trained or loaded
    
    def create_model(self, config):
        if self.architecture == "rnn":
            return LSTM(self.num_classes, 
                self.input_size, 
                config['hidden_dim'], 
                config['num_layers'],
                config['dropout'])
        elif self.architecture == "transformer":
            return TimeSeriesTransformer(self.num_classes, 
                self.input_size, 
                config['num_heads'],
                config['num_layers'],
                config['hidden_dim'],
                dropout=config['dropout'])
        elif self.architecture == 'dense':
            return DenseModel(self.input_size, 
                              config['hidden_dim'],
                              self.num_classes,
                              config['num_layers'],
                              config['dropout'])
            
    def load(self, hyperparameter_info, model_fs):
        self.best_config = hyperparameter_info
        self.model = self.create_model(self.best_config)
        state_dict = model_fs.read_file("model.pth")
        self.model.load_state_dict(state_dict['model'])
        self.data_normalizer.normalization_spec = state_dict['normalizer']
            
    def save(self, model_fs):
        logging.info(f"Saving model to {model_fs}")
        model_fs.write_file({
            'model': self.model.state_dict(),
            'normalizer': self.data_normalizer.normalization_spec
        }, "model.pth")
     
    def get_hyperparameters(self):
        return self.best_config
    
    def pad_collate(self,batch):
        arrays_to_pad = list(zip(*batch))
        x_lens = [len(x) for x in arrays_to_pad[0]]
        padded_arrays = [pad_sequence(xx,batch_first=True,padding_value=0) for xx in arrays_to_pad]
        return (*padded_arrays,torch.LongTensor(x_lens)) 
    
    def create_dataset(self, X, y, ids, fit_normalization=False):
        if fit_normalization:
            self.data_normalizer.fit(X)
        transformed_X = self.data_normalizer.transform(X)
        transformed_y = y.values if isinstance(y, pd.Series) else y
        return TrajectoryDataset(transformed_X, transformed_y, ids)
        
    def loss_fn(self,outputs,targets,lengths):

        if self.model_type == "binary_classification":
            loss_helper = nn.BCEWithLogitsLoss(reduction='none')
        elif self.model_type == "multiclass_classification":
            def loss_helper(outputs, targets):
                return F.cross_entropy(outputs.reshape(-1, outputs.shape[-1]), targets.flatten().long(), reduction='none').reshape(*outputs.shape[:-1])
        elif self.model_type == "regression":
            loss_helper = nn.MSELoss(reduction='none')

        loss = loss_helper(outputs.squeeze(-1) if outputs.shape[-1] == 1 else outputs, targets)
        L = torch.max(lengths).item()
        loss_mask = torch.arange(L)[None, :] < lengths[:, None]
        loss_masked = loss.where(loss_mask, torch.tensor(0.0))
        overall_loss = loss_masked.sum() / (loss_mask.sum() + 1e-3)
        return overall_loss

    def train_func(self, model, optimizer, train_dataloader, val_dataloader, progress_fn=None):
        model.train()
        train_loss = 0
        for i, (features, targets, lengths) in enumerate(train_dataloader):
            outputs = model(features)  # Forward pass
            optimizer.zero_grad()  # Calculate the gradient, manually setting to 0
        
            loss = self.loss_fn(outputs,targets,lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if progress_fn is not None and i % 100 == 99:
                progress_fn({'message': f'training loss {train_loss / (i + 1):.4f}'})
        train_loss /= len(train_dataloader)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for i, (val_features, val_targets, val_len) in enumerate(val_dataloader):
                val_outputs = model(val_features)
                loss = self.loss_fn(val_outputs,val_targets,val_len)
                val_total_loss += loss.item()
                if progress_fn is not None and i % 100 == 99:
                    progress_fn({'message': f'training loss {val_total_loss / (i + 1):.4f}'})
        val_loss = val_total_loss / len(val_dataloader)
        
        return loss,val_loss
    
    def predict(self, test_X, test_ids):
        test_dataset = self.create_dataset(test_X, np.zeros(len(test_X)), test_ids)
        dataloader = DataLoader(test_dataset, 
                                batch_size=self.best_config['batch_size'] if self.best_config else 32,
                                collate_fn=self.pad_collate)
        predict = []
        for i, (features, targets ,lengths) in enumerate(dataloader):
            outputs = self.model(features)
            if self.model_type == 'binary_classification':
                outputs = F.sigmoid(outputs)
            elif self.model_type == 'multiclass_classification':
                outputs = F.softmax(outputs, 2)
            # TODO normalize regression outputs
            for x, l in zip(outputs, lengths):
                predict.append(x[:l].detach().numpy())
        return np.concatenate(predict)
    
    def explain(self, test_X, test_ids, num_batches=None):
        test_dataset = self.create_dataset(test_X, np.zeros(len(test_X)), test_ids)
        dataloader = DataLoader(test_dataset, 
                                batch_size=self.best_config['batch_size'] if self.best_config else 32,
                                collate_fn=self.pad_collate,
                                shuffle=num_batches is not None)
        background = next(iter(dataloader))[0]
        explainer = shap.DeepExplainer(self.model, background)
        shap_values = []
        for i, batch in enumerate(dataloader):
            target, _, lengths = batch
            shaps = explainer.shap_values(target)
            for x, l in zip(shaps, lengths):
                size = l.item()
                trunc_x = x[:size, :]
                flat_x = trunc_x.flatten()
                x_array = flat_x.detach().numpy()
                shap_values.append(x_array)
            if num_batches is not None and i >= num_batches:
                break
        return np.concatenate(shap_values)
    
    def run_with_tuner(self, config, data=None):
        train_X, train_y, train_ids, val_X, val_y, val_ids = data
        nn = NeuralNetwork(self.model_type, 
                           self.architecture, 
                           {k: { "type": "fix", "value": config[k] } for k in config}, 
                           self.input_size, 
                           num_classes=self.num_classes, 
                           tune_reporting=True)
        nn.train(train_X, train_y, train_ids, val_X, val_y, val_ids)
    
    def train(self, train_X, train_y, train_ids, val_X, val_y, val_ids, progress_fn=None, use_tuner=False):
        needs_tune = any(x["type"] != "fix" for x in self.config.values())
        
        if needs_tune and use_tuner:
            
            tune_config = {}
            for key,value in list(self.config.items()):
                keytype = self.config[key]['type']
                if keytype == 'uniform':
                    tune_config[key] = tune.randint(value['value'][0],value['value'][1])
                elif keytype == 'grid search':
                    tune_config[key] = tune.choice(value['value'])
                elif keytype == 'log uniform':
                    tune_config[key] = tune.loguniform(value['value'][0],value['value'][1])
                elif keytype == 'fix':
                    tune_config[key] = value['value']
                else:
                    raise ValueError(f"Unknown hyperparameter type {keytype}")

            ray.shutdown()
            # use local mode to prevent issues with asynchronous file I/O in flask server
            ray.init(logging_level=logging.DEBUG, local_mode=True)

            tuner = Tuner(
                tune.with_parameters(self.run_with_tuner,
                                     data=(train_X, train_y, train_ids, val_X, val_y, val_ids)),
                param_space=tune_config,
                run_config=train.RunConfig(
                    name='test_experiment',
                    checkpoint_config=train.CheckpointConfig(
                        checkpoint_score_attribute='val_loss',
                        num_to_keep=5
                    )
                ),
                tune_config=tune.TuneConfig(
                    num_samples=8,
                    search_alg=HyperOptSearch(
                        metric='val_loss', 
                        mode='min'
                    )
                )
            )

            try:
                tuning_results = tuner.fit()
                best_result = tuning_results.get_best_result(metric='val_loss',mode='min')
                self.best_config = best_result.config
                logging.info('best config:',self.best_config)
                
                with best_result.checkpoint.as_directory() as checkpoint_dir:
                    state_dict = torch.load(os.path.join(checkpoint_dir, 'model.pth'))
                
                    self.model = self.create_model(self.best_config)
                    self.model.load_state_dict(state_dict['model'])
                    self.data_normalizer.normalization_spec = state_dict['normalizer']
            except Exception as e:
                ray.shutdown()
                traceback.print_exc()
                raise RuntimeError(f"Error during hyperparameter search: {e}")
            else:
                ray.shutdown()
        else:
            config_to_use = {k: self.default_config[k] if self.config[k]["type"] != "fix" else self.config[k]["value"]
                                for k in self.config}
            self.best_config = {**config_to_use}
            
            # all parameters are fixed, so we can create the model
            if self.model is None:
                self.model = self.create_model(config_to_use)
                    
            train_dataset = self.create_dataset(train_X, train_y, train_ids, fit_normalization=True)
            val_dataset = self.create_dataset(val_X, val_y, val_ids)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=config_to_use['lr'], weight_decay=0.01)

            train_dataloader = DataLoader(train_dataset,
                                          batch_size=config_to_use['batch_size'],
                                          collate_fn=self.pad_collate)
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config_to_use['batch_size'],
                                        collate_fn=self.pad_collate)

            epochs = config_to_use['num_epochs']
                
            for i in range(epochs):
                train_loss, val_loss = self.train_func(self.model, 
                                                       optimizer, 
                                                       train_dataloader, 
                                                       val_dataloader, 
                                                       progress_fn=(lambda info: progress_fn({'message': f"Epoch {i + 1}, {info['message']}"})) if progress_fn is not None else None)

                if self.tune_reporting:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        # This saves the model to the trial directory
                        torch.save(
                            { 'model': self.model.state_dict(), 'normalizer': self.data_normalizer.normalization_spec },
                            os.path.join(temp_checkpoint_dir, "model.pth")
                        )
                        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                        print("Reporting val loss", val_loss)
                        # Send the current training result back to Tune
                        train.report({"val_loss": val_loss}, checkpoint=checkpoint)
                
            # self.model = model
    
    def evaluate(self, spec, full_metrics, variables, outcomes, ids, train_mask, val_mask, test_mask, progress_fn=None):
        test_pred = self.predict(variables[test_mask], ids[test_mask])
        test_y = outcomes[test_mask]
        # test_y = self.test_y
        metrics = {}
        metrics["labels"] = make_series_summary(test_y 
                                                     if self.model_type != 'multiclass_classification' 
                                                     else pd.Series([spec["output_values"][int(i)] for i in test_y]))

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
                    metrics["predictions"] = make_series_summary(pd.Series([spec["output_values"][int(i)] for i in max_predictions]))

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

        # print('shap values')
        # if progress_fn is not None:
        #     progress_fn({'message': 'Calculating feature importances'})
        # try:
        #     shap_values = self.explain(variables[test_mask], ids[test_mask], num_batches=10)
        #     perf = np.mean(shap_values,axis=0)
        #     perf_std = np.std(shap_values,axis=0)
        #     sorted_perf_index = np.flip(np.argsort(perf))
        #     metrics['feature_importances'] = [{'feature': variables.columns[i], 'mean': perf[i], 'std': perf_std[i]} 
        #                                      for i in sorted_perf_index]
        # except:
        #     logging.error(traceback.format_exc())
        #     print("Error calculating shap values")
        
        # batch = next(iter(self.test_dataloader))
        # target,label,l = batch
        # background = target[:int(self.config['batch_size']/2)]
        # test = target[int(self.config['batch_size']/2):int(self.config['batch_size']/2)+1]
        # explainer = shap.GradientExplainer(self.model, background)
        # shap_values = np.abs(explainer.shap_values(test))

        # # perf = np.mean(shap_values,axis=0)
        # # perf_std = np.std(shap_values,axis=0)
        # # sorted_perf_index = np.flip(np.argsort(perf))
        # # metrics['feature_importances'] = [{'feature': self.column_names[i], 'mean': perf[i], 'std': perf_std[i]} 
        # #                                   for i in sorted_perf_index]
        
        # # TODO fix this
        # perf_list = [np.sum(np.sum(i,axis=0),axis=0) for i in shap_values]
        # perf = np.sum(perf_list,axis=0)
        # sorted_perf_index = np.argsort(perf)
        # performance = [self.column_names[i] for i in sorted_perf_index]

        # metrics['feature_importances'] = [{'feature': self.column_names[i], 'mean': perf[i], 'std': 0} 
        #                                   for i in sorted_perf_index]

        return metrics
    
    # def showMetrics(self):
    #     y_test = self.labels
    #     y_probs = self.get_predicts(self.model,self.test_dataloader)
    #     precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_probs)
    #     f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    #     average_precision = average_precision_score(y_test, y_probs)

    #     plt.figure(figsize=(8, 6))
    #     plt.plot(recalls, precisions, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title('Precision-Recall Curve')
    #     plt.legend(loc='best')
    #     plt.grid(True)
    #     plt.show()

    #     roc_auc,fpr,tpr,thresholds = self.test_func(self.model, self.test_dataloader)
    #     plt.figure()
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc="lower right")
    #     plt.show()
       