from torch import nn
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from typing import Type
from dataclasses import dataclass, field

@dataclass
class KFoldCV:
  X: torch.Tensor
  y: torch.Tensor
  Model: Type[nn.Module]
  model_args: tuple = tuple()
  model_kwargs: dict = field(default_factory=lambda : {}) # field는 dict, list 사용할 때 방법
  epochs: int = 500
  criterion: callable = F.mse_loss
  Optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
  optim_kwargs: dict = field(default_factory=lambda : {})
  trn_dl_kwargs: dict = field(default_factory=lambda : {'batch_size': 32, 'shuffle': True})
  val_dl_kwargs: dict = field(default_factory=lambda : {'batch_size': 32})
  n_splits: int = 5
  metric: torchmetrics.Metric = torchmetrics.MeanSquaredError(squared=False)
  device: str = 'cuda'

  def run(self):
    from torch.utils.data import TensorDataset
    from sklearn.model_selection import KFold
    from tqdm.auto import trange
    from train import train_one_epoch

    model = self.Model(*self.model_args, **self.model_kwargs).to(self.device)
    models = [self.Model(*self.model_args, **self.model_kwargs).to(self.device) for _ in range(self.n_splits)]
    for m in models:# W, b의 값을 동일 한 초기 값을 넣기 위해 큰 의미는 없음 메모리 낭비가 있기에 작은 데이터에서 사용예시
      m.load_state_dict(model.state_dict()) 
    kfold = KFold(n_splits=self.n_splits, shuffle=False)

    metrics = {'trn_rmse': [], 'val_rmse': []}
    print(model)
    for i, (trn_idx, val_idx) in enumerate(kfold.split(self.X)):
      X_trn, y_trn = self.X[trn_idx], self.y[trn_idx]
      X_val, y_val = self.X[val_idx], self.y[val_idx]

      ds_trn = TensorDataset(X_trn, y_trn)
      ds_val = TensorDataset(X_val, y_val)

      dl_trn = DataLoader(ds_trn, **self.trn_dl_kwargs)
      dl_val = DataLoader(ds_val, **self.val_dl_kwargs)
      

      m = models[i]
      optim = self.Optimizer(m.parameters(), **self.optim_kwargs)

      pbar = trange(self.epochs) #trange Tqdm + range
      for _ in pbar:
        train_one_epoch(m, self.criterion, optim, dl_trn, self.metric, self.device)
        trn_rmse = self.metric.compute().item()
        self.metric.reset()
        evaluate(m, dl_val, self.metric, self.device)
        val_rmse = self.metric.compute().item()
        self.metric.reset()
        pbar.set_postfix(trn_rmse=trn_rmse, val_rmse=val_rmse)
      metrics['trn_rmse'].append(trn_rmse)
      metrics['val_rmse'].append(val_rmse)
    return pd.DataFrame(metrics)

def get_args_parser_preprocess(add_help=True):
  import argparse

  parser = argparse.ArgumentParser(description="Data preprocessing", add_help=add_help)
  # inputs
  parser.add_argument("--file-name", default="../../data/wine/winequality-red.csv", type=str, help="data csv file")
  parser.add_argument("--train-csv", default="./data/train.csv", type=str, help="train data csv file")
  parser.add_argument("--test-csv", default="./data/test.csv", type=str, help="test data csv file")
  parser.add_argument("--folder-path", default="./data", type=str, help="diretory path")
  # outputs
  parser.add_argument("--output-train-feas-csv", default="./data/trn_X.csv", type=str, help="output train features")
  parser.add_argument("--output-test-feas-csv", default="./data/tst_X.csv", type=str, help="output test features")
  parser.add_argument("--output-train-target-csv", default="./data/trn_y.csv", type=str, help="output train targets")
  # options
  parser.add_argument("--target-col", default="quality", type=str, help="target column")
  parser.add_argument("--drop-cols", default=None, type=list, help="drop columns")
  parser.add_argument("--fill-num-strategy", default=None, type=str, help="numeric column filling strategy (mean, min, max)")

  return parser

def get_args_parser_train(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

def evaluate(
  model:nn.Module,
  data_loader:DataLoader,
  metric:torchmetrics.metric.Metric,
  device:str='cuda',
) -> None:
  '''evaluate
  
  Args:
      model: model
      data_loader: data loader
      device: device
      metrcis: metrics
  '''
  model.eval()
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      metric.update(output, y)
      
def train_one_epoch(
  model:nn.Module,
  criterion:callable,
  optimizer:torch.optim.Optimizer,
  data_loader:DataLoader,
  metric:torchmetrics.Metric,
  device:str
) -> None:
  '''train one epoch
  
  Args:
      model: model
      criterion: loss
      optimizer: optimizer
      data_loader: data loader
      device: device
  '''
  model.train()
  for X, y in data_loader:
    X, y = X.to(device), y.to(device)
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    metric.update(output, y)