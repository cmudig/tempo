import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch #pytorch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import os
import tempfile
import json
import math
import logging
import sys

import shap
from .raytuner import RayTuner
from .pytorchmodels import LSTM,TimeSeriesTransformer

from sklearn.metrics import f1_score, roc_curve, roc_auc_score, auc, precision_score, recall_score, precision_recall_curve, confusion_matrix, average_precision_score

from .utils import make_series_summary

class Custom3dDataset(Dataset):
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        sample = self.data[idx]
        label = self.labels[idx].astype(np.int64)
        # print(f'label type{label.dtype}')
        # print(f'label {label} type {type(label)}')
        # print(f'label[0] {label[0]} type {type(label[0])}')
        # print(f'label[0][0] {label[0][0]} type{type(label[0][0])}')

        return torch.tensor(sample,dtype=torch.float32),torch.tensor(label,dtype=torch.float32)

class CustomDataset:
    def __init__(self,x,y,ids):
        self.dataset = self.process_data(x,y,ids)
    
    def filter(self,x,y,ids):
        concat = pd.concat([ids,x,y], axis=1)
        filter_data = concat[concat['purchase'].notna()]

        id_counts = filter_data['user_id'].value_counts()
        # print(f'data before filter {filter_data}')
        filtered_data = filter_data
        # filtered_data = filter_data[filter_data['user_id'].isin(id_counts[(id_counts>=2)&(id_counts<=150)].index)]
        # print(f'data after filter {filtered_data}')
        print('nan number',filtered_data.isna().sum().sum())
        # filtered_data = filtered_data.dropna()

        return filtered_data
    
    def scale(self,filtered_data):
        ss = StandardScaler()
        
        to_scale = filtered_data.iloc[:,1:-1]
        targets = filtered_data.iloc[:,-1:]
        indexs = filtered_data.iloc[:,0:1]
        scaled = ss.fit_transform(to_scale)
        scaled_df = pd.DataFrame(scaled)
        scaled_df.reset_index(drop=True,inplace=True)
        indexs.reset_index(drop=True,inplace=True)
        targets.reset_index(drop=True,inplace=True)
        scaled_data = pd.concat([indexs,scaled_df,targets],axis=1)

        return scaled_data
    
    def convert_to_numpy(self,scaled):
        grouped = scaled.groupby('user_id')
        x_seq = []
        y_seq = []
        for key,group in grouped:
            x = group.iloc[:,1:-1]
            y = group.iloc[:,-1:]
            x_n = x.to_numpy()
            y_n = y.to_numpy()
            x_seq.append(x_n)
            y_seq.append(y_n)
        return x_seq,y_seq

    def process_data(self, x, y, ids):
        filtered = self.filter(x, y, ids)
        scaled = self.scale(filtered)
        x_seq,y_seq = self.convert_to_numpy(scaled)
        dataset = Custom3dDataset(x_seq, y_seq)
        return dataset

class NeuralNetwork:
    def __init__(self,model_type,train_X,train_y,train_ids,test_X,
                    test_y,test_ids,val_X,val_y,val_ids,config):

        tmp_config = {}
        default_config = {
            'num_epochs': 1,
            'batch_size': 64,
            'lr': 0.001,
            'num_layers': 1,

            # lstm parameters
            'num_classes': 1,
            'hidden_size': 2,
            
            # transformer parameters
            'num_heads' : 4,
            'hidden_dim': 128,
        }

        self.tuning_config = config

        for key,value in list(config.items()):
            if config[key]["type"] == 'fix':
                tmp_config[key] = value['value']
            else:
                tmp_config[key] = default_config[key]
        
        self.config = tmp_config

        self.model_type = model_type
        self.batch_size = self.config['batch_size']
        self.input_size = self.config['input_size']
        self.num_layers = self.config['num_layers']

        self.train_instances = len(train_X)
        self.train_trajectories = len(np.unique(train_ids))
        self.test_instances = len(test_X)
        self.test_trajectories = len(np.unique(test_ids))
        self.val_instances = len(val_X)
        self.val_trajectories = len(np.unique(val_ids))

        self.column_names = list(train_X.columns)

        self.train_y = train_y.values
        self.test_y = test_y.values
        self.val_y = val_y.values

        train_data = CustomDataset(train_X,train_y,train_ids)
        test_data = CustomDataset(test_X,test_y,test_ids)
        val_data = CustomDataset(val_X,val_y,val_ids)

        self.data = {
            'train': train_data.dataset,
            'test': test_data.dataset,
            'val': val_data.dataset
        }

        self.train_dataloader = DataLoader(self.data['train'],batch_size = self.batch_size,collate_fn=self.pad_collate)
        self.test_dataloader = DataLoader(self.data['test'],batch_size = self.batch_size,collate_fn=self.pad_collate)
        self.val_dataloader = DataLoader(self.data['val'],batch_size = self.batch_size,collate_fn=self.pad_collate)

        if(self.model_type == 'rnn'):

            self.num_classes = self.config['num_classes']
            self.hidden_size = self.config['hidden_size']

            self.model = LSTM(self.num_classes,self.input_size,self.hidden_size,self.num_layers)
        
        elif(self.model_type == 'transformer'):

            self.num_heads = self.config['num_heads']
            self.hidden_dim = self.config['hidden_dim']

            self.model = TimeSeriesTransformer(self.input_size, self.num_heads, self.num_layers, self.hidden_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.config['lr'])
    
    def pad_collate(self,batch):
        arrays_to_pad = list(zip(*batch))
        x_lens = [len(x) for x in arrays_to_pad[0]]
        padded_arrays = [pad_sequence(xx,batch_first=True,padding_value=0) for xx in arrays_to_pad]
        return (*padded_arrays,torch.LongTensor(x_lens)) 
    
    def loss_fn(self,outputs,targets,lengths):

        loss_helper = nn.BCEWithLogitsLoss(reduction='none')

        loss = loss_helper(outputs, targets)
        loss = torch.squeeze(loss,dim=-1)
        L = torch.max(lengths).item()
        loss_mask = torch.arange(L)[None, :] < lengths[:, None]
        loss_masked = loss.where(loss_mask, torch.tensor(0.0))
        overall_loss = loss_masked.sum() / loss_mask.sum()
        return overall_loss

    def train_func(self,model,optimizer,train_dataloader,val_dataloader):
        model.train()
        for i, (features, targets, lengths) in enumerate(train_dataloader):
            outputs = model(features)  # Forward pass
            optimizer.zero_grad()  # Calculate the gradient, manually setting to 0
            
            loss = self.loss_fn(outputs,targets,lengths)
            loss.backward()
            optimizer.step()

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for _ ,(val_features, val_targets,val_len) in enumerate(val_dataloader):
                val_outputs = model(val_features)
                val_total_loss += self.loss_fn(val_outputs,val_targets,val_len)
        val_loss = val_total_loss / len(val_dataloader)
        print(f'training loss: {loss}, val loss: {val_loss}')
        return loss,val_loss
    
    def predict(self,data_type=None,model=None):
        if(model is None):
            model = self.model
            
        if data_type == 'val':
            dataloader = self.val_dataloader
        elif data_type == 'train':
            dataloader = self.train_dataloader
        else:
            dataloader = self.test_dataloader
        
        predict = np.array([])
        for i, (features, targets ,lengths) in enumerate(dataloader):
            outputs = model(features)
            outputs = F.sigmoid(outputs)
            for x,y,l in zip(outputs, targets, lengths):
                size = l.item()
                trunc_x = x[:size, :]
                flat_x = trunc_x.flatten()
                x_array = flat_x.detach().numpy()
                predict = np.concatenate([predict,x_array])
        return predict
    
    def getlabels(self,data_type=None,model=None):
        if(model is None):
            model = self.model

        if data_type == 'val':
            dataloader = self.val_dataloader
        elif data_type == 'train':
            dataloader = self.train_dataloader
        else:
            dataloader = self.test_dataloader

        labels = np.array([])
        for i, (features, targets ,lengths) in enumerate(dataloader):
            outputs = model(features)
            outputs = F.sigmoid(outputs)
            for x,y,l in zip(outputs, targets, lengths):
                size = l.item()
                trunc_y = y[:size, :]
                flat_y = trunc_y.flatten()
                y_array = flat_y.detach().numpy()
                labels = np.concatenate([labels,y_array])

        return labels
    
    def test_func(self,model,data_type=None):

        predict = self.predict(data_type,model)
        labels = self.getlabels(data_type,model)
        fpr, tpr, thresholds = roc_curve(labels, predict)
        roc_auc = auc(fpr, tpr)

        return roc_auc,fpr,tpr,thresholds
    
    def train(self,config=None,tuning_mode=True):
        
        if(config is None):
            config = self.config
        
        if(tuning_mode):
            tuner = RayTuner(self.model_type,self.tuning_config)
            best_result = tuner.fit(self.data)
            best_config = best_result.config
            print('best config',best_config)
            print('best result',best_result)
            self.train_dataloader = DataLoader(self.data['train'],batch_size = best_config['batch_size'],collate_fn=self.pad_collate)
            self.test_dataloader = DataLoader(self.data['test'],batch_size = best_config['batch_size'],collate_fn=self.pad_collate)
            self.val_dataloader = DataLoader(self.data['val'],batch_size = best_config['batch_size'],collate_fn=self.pad_collate)
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=best_config['lr'])

            if not hasattr(best_result, 'checkpiont'):
                self.train(best_config,False)
            else:
                with best_result.checkpoint.as_directory() as checkpoint_dir:
                    state_dict = torch.load(os.path.join(checkpoint_dir, 'model.pth'))
                
                    if(self.model_type == 'rnn'):
                        model = LSTM(self.num_classes,self.input_size,best_config['hidden_size'],best_config['num_layers'])
                    elif(self.model_type == 'transformer'):
                        model = TimeSeriesTransformer(self.input_size, self.num_heads, self.num_layers, self.hidden_dim)
                    
                    model.load_state_dict(state_dict)
                    
                self.model = model
        
        else:
            print('no checkpoint training')
            epochs = config['num_epochs']
                
            if(self.model_type == 'transformer'):
                model = TimeSeriesTransformer(self.input_size, config['num_heads'], config['num_layers'], config['hidden_dim'])
            else:
                model = LSTM(self.num_classes,self.input_size,config['hidden_size'],config['num_layers'])
            
            early_stop = 3
            counter = 0
            best_val_loss = float('inf')
            for i in range(epochs):
                train_loss,val_loss = self.train_func(model,self.optimizer,self.train_dataloader,self.val_dataloader)

                if(val_loss < best_val_loss):
                    best_val_loss = val_loss
                    counter = 0
                else:
                    early_stop += 1
                    if counter >= early_stop:
                        print('Early Stopping')
                        break
                
                roc_auc,fpr,tpr,thresholds = self.test_func(model,'test')
                print(f'roc_auc {roc_auc}')
                
            self.model = model
    
    def evaluate(self,spec,full_metrics,variables,outcomes,train_mask,val_mask,test_mask,row_mask,**model_params):
        test_pred = self.predict(data_type='test')
        test_y = self.getlabels(data_type='test')
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
        
        metrics["n_train"] = {"instances": self.train_instances, "trajectories": self.train_trajectories}
        metrics["n_val"] = {"instances": self.val_instances, "trajectories": self.val_trajectories}
        metrics["n_test"] = {"instances": self.test_instances, "trajectories": self.test_trajectories}

        print('shap values')
        batch = next(iter(self.test_dataloader))
        target,label,l = batch
        background = target[:int(self.batch_size/2)]
        test = target[int(self.batch_size/2):int(self.batch_size/2)+1]
        explainer = shap.GradientExplainer(self.model, background)
        shap_values = explainer.shap_values(test)

        # TODO fix this
        perf_list = [np.sum(np.sum(i,axis=0),axis=0) for i in shap_values]
        perf = np.sum(perf_list,axis=0)
        sorted_perf_index = np.argsort(perf)
        performance = [self.column_names[i] for i in sorted_perf_index]

        metrics['features'] = performance
        metrics['shap values'] = perf

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
       