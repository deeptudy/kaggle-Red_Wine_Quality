import torch
import torch.nn.functional as F
import torchmetrics
from nn import ANN

config = {
  ### Input and Output files ###
  'files': {
    ''
    'X_csv': './data/trn_X.csv',
    'y_csv': './data/trn_y.csv',
    'tst_csv': './data/tst_X.csv',
    'output': './model.pth',
    'output_csv': './results/five_fold.csv',
  },
  ### Model ###
  'model': ANN,
  'model_params': {
    'input_dim': 'auto', # Always will be determined by the data shape
    'hidden_dim': [128, 64, 32],
    'use_dropout': True,
    'drop_ratio': 0.3,
    'activation': 'relu',
  },
  ### Train Parameters
  'train_params': {
    ### DataLoader
    'data_loader_params': {
      'batch_size': 32,
      'shuffle': True,
    },
    ### Loss (Cost, Criterion)
    'loss': F.mse_loss,
    ### Optimizer
    'optim': torch.optim.Adam,
    'optim_params': {
      'lr': 0.0001,
    },
    ### Metric, RMSE
    'metric': torchmetrics.MeanSquaredError(squared=False),
              
    ### Device
    'device': 'cpu',
    ### Epochs
    'epochs': 1000,
  },
  ### Cross Validation
  'cv_params':{
    'n_split': 5,
  },

}