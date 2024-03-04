from utils import get_args_parser_preprocess
import pandas as pd
from typing import Literal
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import os

@dataclass # 자동으로 __init__을 만들어줌
class WineData:
  file_name: str = '../../data/wine/winequality-red.csv'
  file_trn: str = './data/train.csv'
  file_tst: str = './data/test.csv'
  target_col: str = 'quality'
  drop_cols: tuple[str] = None
  fill_num_strategy: Literal['mean', 'min', 'max'] = None # 전처리 기능을 만들어 쉽게 테스트 가능
  folder_path: str = './data'

  # 폴더가 존재하지 않으면 생성
  def _make_dir(self):
    try:
        os.mkdir(self.folder_path)
        print(f"폴더 '{self.folder_path}'가 생성되었습니다.")
    except FileExistsError:
        print(f"폴더가 있습니다.")
  
  def _split_data(self):
        
        df = pd.read_csv(self.file_name)
        
        trn_df, tst_df = train_test_split(df, test_size=0.2, random_state=2023)
        trn_df.to_csv('./data/train.csv', index=False)
        tst_df.to_csv('./data/test.csv', index=False)
        
        return trn_df, tst_df

  def _read_df(self, split:Literal['train', 'test']='train'):
    if split == 'train': # target에 nan값이 있으면 drop
      df = pd.read_csv(self.file_trn)
      df.dropna(axis=0, subset=[self.target_col], inplace=True)
      target = df[self.target_col]
      df.drop([self.target_col], axis=1, inplace=True)
      return df, target
    elif split == 'test':
      df = pd.read_csv(self.file_tst)
      df.drop([self.target_col], axis=1, inplace=True)
      return df
    raise ValueError(f'"{split}" is not acceptable.')

  def preprocess(self):
    if not os.path.exists(self.file_name):
      raise FileNotFoundError(f"{self.file_name} 파일이 존재하지 않습니다.")
    
    self._make_dir()
    trn_df, tst_df = self._split_data()
    
    trn_df, target = self._read_df('train')
    tst_df = self._read_df('test')
    
    # drop `drop_cols`
    if self.drop_cols is not None:
      trn_df.drop(self.drop_cols, axis=1, inplace=True)
      tst_df.drop(self.drop_cols, axis=1, inplace=True)

    # fill the numerical columns using `fill_num_strategy`
    if self.fill_num_strategy == 'mean':
      fill_values = trn_df.mean(axis=1)
    elif self.fill_num_strategy == 'min':
      fill_values = trn_df.min(axis=1)
    elif self.fill_num_strategy == 'max':
      fill_values = trn_df.max(axis=1)
    else:
      fill_values = trn_df
  
    trn_df.fillna(fill_values, inplace=True)
    tst_df.fillna(fill_values, inplace=True)
    
    return trn_df, target, tst_df

if __name__ == "__main__":
  args = get_args_parser_preprocess().parse_args()
  wine_data = WineData(
    args.file_name,
    args.train_csv,
    args.test_csv,
    args.target_col,
    args.drop_cols,
    args.fill_num_strategy,
    args.folder_path
  )
  trn_X, trn_y, tst_X = wine_data.preprocess()
  print(trn_X.shape, trn_y.shape, tst_X.shape)
  trn_X.to_csv(args.output_train_feas_csv)
  tst_X.to_csv(args.output_test_feas_csv)
  trn_y.to_csv(args.output_train_target_csv)