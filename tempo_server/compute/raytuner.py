import ray
from ray import train, tune
from ray.tune import Tuner
from ray.train import Checkpoint,Result
from ray.tune.schedulers import ASHAScheduler

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from .pytorchmodels import LSTM, TimeSeriesTransformer

import tempfile
import logging
import os
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

class RayTuner:
    def __init__(self,model_type,config):

        # config['lr'] = tune.loguniform(config['lr'][0],config['lr'][1])
        # config['batch_size'] = tune.choice(config['batch_size'])
        # config['num_layers'] = tune.randint(config['num_layers'][0],config['num_layers'][1])
        # config['hidden_size'] = tune.randint(config['hidden_size'][0],config['hidden_size'][1])

        self.model_type = model_type

        default_config = {
            "batch_size": tune.choice([32,64]),
            "num_layers": tune.randint(0,5),
            "hidden_size": tune.randint(1,5),
            "lr": tune.loguniform(1e-4,1e-2),
            "num_heads": tune.randint(0,4),
            "hidden_dim": tune.choice([128,256])
        }

        tmp_config = {}
        for key,value in list(config.items()):
            keytype = config[key]['type']
            if keytype == 'uniform':
                tmp_config[key] = tune.randint(value['value'][0],value['value'][1])
            elif keytype == 'grid_search':
                tmp_config[key] = tune.choice(value['value'])
            elif keytype == 'log_uniform':
                tmp_config[key] = tune.loguniform(value['value'][0],value['value'][1])
            elif keytype == 'fix':
                tmp_config[key] = value['value']
            else:
                tmp_config[key] = default_config[key]

        self.config = tmp_config

    def loss_fn(self,outputs,targets,lengths):

        loss_helper = nn.BCEWithLogitsLoss(reduction='none')

        loss = loss_helper(outputs, targets)
        loss = torch.squeeze(loss,dim=-1)
        L = torch.max(lengths).item()
        loss_mask = torch.arange(L)[None, :] < lengths[:, None]
        loss_masked = loss.where(loss_mask, torch.tensor(0.0))
        overall_loss = loss_masked.sum() / loss_mask.sum()
        return overall_loss
    
    def train_func(self,model,optimizer,dataloader):
        model.train()
        for i, (features, targets, lengths) in enumerate(dataloader):
            outputs = model(features)  # Forward pass
            optimizer.zero_grad()  # Calculate the gradient, manually setting to 0
            
            loss = self.loss_fn(outputs,targets,lengths)
            loss.backward()
            optimizer.step()

        print(f'training loss: {loss}')
        return loss
    
    def pad_collate(self,batch):
        arrays_to_pad = list(zip(*batch))
        x_lens = [len(x) for x in arrays_to_pad[0]]
        padded_arrays = [pad_sequence(xx,batch_first=True,padding_value=0) for xx in arrays_to_pad]
        return (*padded_arrays,torch.LongTensor(x_lens))
    
    def test_func(self,model,dataloader):
        predict = np.array([])
        labels = np.array([])
        for i, (features, targets ,lengths) in enumerate(dataloader):
            outputs = model(features)
            outputs = F.sigmoid(outputs)
            for x,y,l in zip(outputs, targets, lengths):
                size = l.item()
                trunc_x = x[:size, :]
                trunc_y = y[:size, :]
                flat_x = trunc_x.flatten()
                flat_y = trunc_y.flatten()
                x_array = flat_x.detach().numpy()
                y_array = flat_y.detach().numpy()
                predict = np.concatenate([predict,x_array])
                labels = np.concatenate([labels,y_array])
        fpr, tpr, thresholds = roc_curve(labels, predict)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def train(self,config,data=None):
        
        epochs = config['num_epochs']
        batch_size = config['batch_size']

        train_dataloader = DataLoader(data['train'],batch_size = batch_size, collate_fn=self.pad_collate)
        test_dataloader = DataLoader(data['test'],batch_size = batch_size, collate_fn=self.pad_collate)

        if self.model_type == 'rnn':
            model = LSTM(config['num_classes'],config['input_size'],config['hidden_size'],config['num_layers'])
        elif self.model_type == 'transformer':
            model = TimeSeriesTransformer(config['input_size'], config['num_heads'], config['num_layers'], config['hidden_dim'])
        optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])

        for i in range(epochs):
            self.train_func(model,optimizer,train_dataloader)
            roc_auc = self.test_func(model,test_dataloader)

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                if (i + 1) % 5 == 0:
                    # This saves the model to the trial directory
                    torch.save(
                        model.state_dict(),
                        os.path.join(temp_checkpoint_dir, "model.pth")
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                # Send the current training result back to Tune
                train.report({"roc_auc": roc_auc}, checkpoint=checkpoint)
    
    def fit(self,data):
        config = self.config

        ray.shutdown()
        ray.init(logging_level=logging.DEBUG)

        tuner = Tuner(
            tune.with_parameters(self.train,data=data),
            param_space = config,

            run_config = train.RunConfig(
                name = 'test_experiement',
                stop = {'training_iteration': 100},
                checkpoint_config = train.CheckpointConfig(
                    checkpoint_score_attribute='roc_auc',
                    num_to_keep=5
                )
            ),
            tune_config = tune.TuneConfig(
                num_samples=1,
                scheduler = ASHAScheduler(
                    metric = 'roc_auc', 
                    mode = 'max'
                )
            )
        )

        tuning_results = tuner.fit()
        best_result = tuning_results.get_best_result(metric='roc_auc',mode='max')
        
        return best_result
    
    def showTuningResults(self):
        ax = None
        for result in self.tuning_results:
            label = f"lr={result.config['lr']:.3f}"
            if ax is None:
                ax = result.metrics_dataframe.plot("training_iteration", "roc_auc", label=label)
            else:
                result.metrics_dataframe.plot("training_iteration", "mean_accuracy", ax=ax, label=label)
        ax.set_title("ROC_AUC vs. Training Iteration for All Trials")
        ax.set_ylabel("ROC_AUC")


