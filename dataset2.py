###### 좀 더 많은 augmentation을 해보자
import os
import numpy as np
import torch
from PIL import Image, ImageEnhance
import imgaug.augmenters as iaa

class Dataset(torch.utils.data.Dataset):
    ##데이터가 존재하는 디렉토리 주소와 변환을 매개변수로 지정.
        def __init__(self, data_dir, transform=None):
            self.data_dir=data_dir
            self.transform=transform    

            lst_data = os.listdir(self.data_dir)

            lst_label = [f for f in lst_data if f.startswith('label')]
            lst_input = [f for f in lst_data if f.startswith('input')]

            lst_label.sort()
            lst_input.sort()

            self.lst_label = lst_label
            self.lst_input = lst_input

        def __len__(self):
            return len(self.lst_input)
        
        def __getitem__(self,index):
            ##원하는 데이터 가져오기
            label = Image.open(os.path.join(self.data_dir, self.lst_label[index]))
            input = Image.open(os.path.join(self.data_dir, self.lst_input[index]))            

            label = np.asarray(label)
            input = np.asarray(input) 

            
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
