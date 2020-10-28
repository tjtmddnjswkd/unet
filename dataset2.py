###### 좀 더 많은 augmentation을 해보자
import os
import numpy as np
import torch
from PIL import Image
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

# ##데이터셋 png파일로 다 저장하기
# for i in range(1,21):
#         img_input = Image.open(os.path.join('/daintlab/home/tmddnjs3467/workspace/vessel/test/images', '%02d_test.tif' % i))
        
#         img_input = img_input.resize((512, 512))

#         input_ = np.asarray(img_input)
#         image = Image.fromarray(input_)
#         image.save('input_%02d.png' % i)
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

            if np.random.rand() > 0.5:
                aug = iaa.MultiplyAndAddToBrightness(mul=0.7)
                input = aug(return_batch = False, image = input)
            if np.random.rand() > 0.5:
                aug = iaa.ChannelShuffle(0.35)
                input = aug(return_batch = False, image = input)
           
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
