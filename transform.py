import torch
import numpy as np 

## 트랜스폼 구현

class ToTensor(object):
  def __call__(self, data):
    label, input = data['label'], data['input']
    ##numpy array는 (w,h,c) 순이기 때문에 바꿔줌
    label = label.transpose((2,0,1)).astype(np.float32)
    input = input.transpose((2,0,1)).astype(np.float32)

    data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

    return data
class Normalization(object):
  def __init__(self, mean=0.5, std=0.5):
    self.mean = mean
    self.std = std
  
  def __call__(self, data):
    label, input = data['label'], data['input']
    ##label은 1과 0 으로 이루어져있으므로 노말라이제이션 안함
    input = (input - self.mean) / self.std

    data = {'label': label, 'input': input}
    
    return data

class RandomFlip(object):
  def __call__(self, data):
    label, input = data['label'], data['input']
    #오른쪽 왼쪽 뒤집기
    if np.random.rand() > 0.5:
      label = np.fliplr(label)
      input = np.fliplr(input)
    #위아래 뒤집기
    if np.random.rand() > 0.5:
       label = np.flipud(label)
       input = np.flipud(input)
    
    data = {'label': label, 'input': input}

    return data
