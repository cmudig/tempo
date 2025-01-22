import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch #pytorch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import os
from scipy.stats import percentileofscore
import tempfile
from ray import train, tune
from ray.tune import Tuner
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import logging
import traceback

import shap
from .raytuner import RayTuner
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
        return np.where(np.isnan(new_data), 0, new_data)

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
                config['hidden_dim']['value'], 
                config['num_layers']['value'],
                config['dropout']['value'])
        elif self.architecture == "transformer":
            return TimeSeriesTransformer(self.num_classes, 
                self.input_size, 
                config['num_heads']['value'],
                config['num_layers']['value'],
                config['hidden_dim']['value'],
                dropout=config['dropout']['value'])
        elif self.architecture == 'dense':
            return DenseModel(self.input_size, 
                              config['hidden_dim']['value'],
                              self.num_classes,
                              config['num_layers']['value'],
                              config['dropout']['value'])
            
    def load(self, hyperparameter_info, model_fs):
        self.best_config = hyperparameter_info
        self.model = self.create_model(self.best_config)
        self.model.load_state_dict(model_fs.read_file("model.pth"))
            
    def save(self, model_fs):
        model_fs.write_file(self.model.state_dict(), "model.pth")
     
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
            loss_helper = nn.CrossEntropyLoss(reduction='none')
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
            for x,y,l in zip(outputs, targets, lengths):
                size = l.item()
                trunc_x = x[:size, :]
                flat_x = trunc_x.flatten()
                x_array = flat_x.detach().numpy()
                predict.append(x_array)
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
    
    def train(self, train_X, train_y, train_ids, val_X, val_y, val_ids, progress_fn=None, use_tuner=False):
        needs_tune = any(x["type"] != "fix" for x in self.config.values())
        
        if needs_tune and use_tuner:
            
            # TODO run the ray tuner directly here, creating instances of the 
            # NeuralNetwork class with fixed parameters for each parameter set being tested
            tuner = RayTuner(self.architecture,self.tuning_config)
            best_result = tuner.fit(self.data)
            best_config = best_result.config
            self.config = best_config
            print('best config',best_config)
            print('best result',best_result)
            logging.info(f'Ray Tuner Config {best_config}')
            # self.train_dataloader = DataLoader(self.data['train'],batch_size = best_config['batch_size'],collate_fn=self.pad_collate)
            # self.test_dataloader = DataLoader(self.data['test'],batch_size = best_config['batch_size'],collate_fn=self.pad_collate)
            # self.val_dataloader = DataLoader(self.data['val'],batch_size = best_config['batch_size'],collate_fn=self.pad_collate)
            # self.optimizer = torch.optim.Adam(self.model.parameters(),lr=best_config['lr'])

            if not hasattr(best_result, 'checkpoint'):
                self.train(best_config,False)
            else:
                with best_result.checkpoint.as_directory() as checkpoint_dir:
                    state_dict = torch.load(os.path.join(checkpoint_dir, 'model.pth'))
                
                    self.model = self.create_model(best_config)                    
                    self.model.load_state_dict(state_dict)

        else:
            config_to_use = {k: {"type": "fix", "value": self.default_config[k]} if self.config[k]["type"] != "fix" else self.config[k]
                                for k in self.config}
            self.best_config = {k: v["value"] for k, v in config_to_use.items()}
            
            # all parameters are fixed, so we can create the model
            if self.model is None:
                self.model = self.create_model(config_to_use)
                    
            train_dataset = self.create_dataset(train_X, train_y, train_ids, fit_normalization=True)
            val_dataset = self.create_dataset(val_X, val_y, val_ids)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=config_to_use['lr']['value'], weight_decay=0.01)

            train_dataloader = DataLoader(train_dataset,
                                          batch_size=config_to_use['batch_size']['value'],
                                          collate_fn=self.pad_collate)
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config_to_use['batch_size']['value'],
                                        collate_fn=self.pad_collate)

            epochs = config_to_use['num_epochs']['value']
                
            early_stop = 3
            counter = 0
            best_val_loss = float('inf')
            for i in range(epochs):
                train_loss, val_loss = self.train_func(self.model, 
                                                       optimizer, 
                                                       train_dataloader, 
                                                       val_dataloader, 
                                                       progress_fn=lambda info: progress_fn({'message': f"Epoch {i + 1}, {info['message']}"}))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= early_stop:
                        print('Early Stopping')
                        break

                if self.tune_reporting:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        checkpoint = None
                        if (i + 1) % 20 == 0:
                            # This saves the model to the trial directory
                            torch.save(
                                self.model.state_dict(),
                                os.path.join(temp_checkpoint_dir, "model.pth")
                            )
                            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

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
                                                     else pd.Series([spec["output_values"][i] for i in test_y]))

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

        print('roc')
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
       