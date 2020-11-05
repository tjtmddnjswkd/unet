import os
import numpy as np
import torch

##데이터셋
class Dataset(torch.utils.data.Dataset):
  ##데이터가 존재하는 디렉토리 주소와 변환을 매개변수로 지정.
  def __init__(self, data_dir, transform=None, type='train'):
    if self.type == 'train':
      self.data_dir=data_dir
      self.transform=transform

      lst_data = os.listdir(self.data_dir)

      lst_label = [f for f in lst_data if f.startswith('label')]
      lst_input = [f for f in lst_data if f.startswith('input')]

      lst_label.sort()
      lst_input.sort()

      self.lst_label = lst_label
      self.lst_input = lst_input
    else: #테스트 데이터에 대해서는 라벨이 존재하지 않음.
      self.data_dir=data_dir
      self.transform=transform

      lst_data = os.listdir(self.data_dir)

      lst_input = [f for f in lst_data if f.startswith('input')]

      lst_input.sort()

      self.lst_input = lst_input

  def __len__(self):
    return len(self.lst_input)
  
  def __getitem__(self,index):
    ##원하는 데이터 가져오기
    label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
    input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

    label = label/255.0
    input = input/255.0

    if label.ndim == 2:
      label = label[:, :, np.newaxis]
    if input.ndim == 2:
      input = input[:, :, np.newaxis]
    
    data = {'input': input, 'label': label}
    ##변환 정의된경우 변환한 데이터를 리턴.
    if self.transform:
      data = self.transform(data)

    return data
