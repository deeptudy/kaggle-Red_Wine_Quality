import torch
import pandas as pd
import os.path
from os import path
from utils import get_args_parser_train


if __name__ == "__main__":
  import numpy as np
  from utils import KFoldCV

  args = get_args_parser_train().parse_args()
  
  exec(open(args.config).read())
  cfg = config

  train_params = cfg.get('train_params')
  device = train_params.get('device')

  files = cfg.get('files')
  X_df = pd.read_csv(files.get('X_csv'),index_col=0)
  y_df = pd.read_csv(files.get('y_csv'),index_col=0)

  X, y = torch.tensor(X_df.to_numpy(dtype=np.float32)), torch.tensor(y_df.to_numpy(dtype=np.float32))

  Model = cfg.get('model')
  model_params = cfg.get('model_params')
  model_params['input_dim'] = X.shape[-1]
  
  dl_params = train_params.get('data_loader_params')

  Optim = train_params.get('optim')
  optim_params = train_params.get('optim_params')

  metric = train_params.get('metric').to(device)
  
  cv = KFoldCV(X, y, Model, model_kwargs=model_params,
               epochs=train_params.get('epochs'),
               criterion=train_params.get('loss'),
               Optimizer=Optim,
               optim_kwargs=optim_params,
               trn_dl_kwargs=dl_params, val_dl_kwargs=dl_params,
               metric=metric,
               device=device)
  res = cv.run()

  res = pd.concat([res, res.apply(['mean', 'std'])])
  print(res)
  if not path.exists('results'):
    os.mkdir("./results")
  res.to_csv(files.get('output_csv'))