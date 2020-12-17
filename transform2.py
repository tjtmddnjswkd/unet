import torch
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
import random
import sys

s = random.randint(1,sys.maxsize)
## 트랜스폼 구현
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        
        t = transforms.ToTensor()
        input = t(input)
        label = t(label)
        
        data = {'label': label, 'input': input}

        return data
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
  
    def __call__(self, data):
        label, input = data['label'], data['input']
        ##label은 1과 0 으로 이루어져있으므로 노말라이제이션 안함
        
        t = transforms.Normalize(self.mean, self.std)
        input = t(input)

        data = {'label': label, 'input': input}
        
        return data

class RandomFlip(object):
    def __call__(self, data):
    
        label, input = data['label'], data['input']
        
        t = transforms.RandomHorizontalFlip(1)
        if random.random() < 0.5:
            input = t(input)
            label = t(label)
        t = transforms.RandomVerticalFlip(1)
        if random.random() < 0.5:
            input = t(input)
            label = t(label)

        data = {'label': label, 'input': input}
        
        return data

class RandomBrightly(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        
        t = transforms.ColorJitter(brightness = 0.5)
        if random.random() < 0.5:
            input = t(input)
        
        data = {'label': label, 'input': input}

        return data
print()
class RandomContrast(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        
        t = transforms.ColorJitter(contrast = 0.5)
        if random.random() < 0.5:
            input = t(input)
        
        data = {'label': label, 'input': input}

        return data

class GaussianBlur(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        t = transforms.GaussianBlur(5, sigma=(0.1, 2.0))
        if random.random() < 0.5:
            input = t(input)

        data = {'label': label, 'input': input}

        return data

class RandomRotate(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        t = transforms.RandomRotation(360)
        torch.manual_seed(s)
        input = t(input)
        torch.manual_seed(s)
        label = t(label)

        data = {'label': label, 'input': input}

        return data
