import os
import numpy as np
import torch
from PIL import Image, ImageEnhance
import imgaug.augmenters as iaa
import random
class Dataset_train(torch.utils.data.Dataset):
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
           #랜덤 칼라
            if np.random.rand() < 0.5:
                random_factor = np.random.randint(0, 31) / 10.  
                color_image = ImageEnhance.Color(input).enhance(random_factor)  
                random_factor = np.random.randint(10, 21) / 10.  
                brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
                random_factor = np.random.randint(10, 21) / 10.  
                contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  
                random_factor = np.random.randint(0, 31) / 10. 
                input = ImageEnhance.Sharpness(contrast_image).enhance(random_factor) 
 
            label = np.asarray(label).copy()
            input = np.asarray(input).copy()

            def gaussianNoisy(im, mean=0.2, sigma=0.3):
                for i in range(len(im)):
                    im[i] += random.gauss(mean, sigma)
                return im

            if np.random.rand() < 0.5:
                width, height = input.shape[:2]
                img_r = gaussianNoisy(input[:, :, 0].flatten())
                img_g = gaussianNoisy(input[:, :, 1].flatten())
                img_b = gaussianNoisy(input[:, :, 2].flatten())
                input[:, :, 0] = img_r.reshape([width, height])
                input[:, :, 1] = img_g.reshape([width, height])
                input[:, :, 2] = img_b.reshape([width, height])
                input = np.uint8(input)
    
            
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

