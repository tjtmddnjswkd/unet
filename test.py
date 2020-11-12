import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np 
from PIL import Image
from unet import UNet
import wandb

################test용 데이터셋
class Dataset(torch.utils.data.Dataset):
  ##데이터가 존재하는 디렉토리 주소와 변환을 매개변수로 지정.
  def __init__(self, data_dir, transform=None):
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
    # input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
    input = Image.open(os.path.join(self.data_dir, self.lst_input[index]))
    
    input = np.asarray(input)

    input = input/255.0

    if input.ndim == 2:
      input = input[:, :, np.newaxis]

    data = {'input' : input}
    ##변환 정의된경우 변환한 데이터를 리턴.
    if self.transform:
      data = self.transform(data)

    return data

## 트랜스폼 구현

class ToTensor(object):
  def __call__(self, data):
    input = data['input']
    ##numpy array는 (w,h,c) 순이기 때문에 바꿔줌
    input = input.transpose((2,0,1)).astype(np.float32)

    data = {'input': torch.from_numpy(input)}

    return data
class Normalization(object):
  def __init__(self, mean=0.5, std=0.5):
    self.mean = mean
    self.std = std
  
  def __call__(self, data):
    input = data['input']
    
    input = (input - self.mean) / self.std

    data = {'input': input}
    
    return data

test_dir = '/daintlab/home/tmddnjs3467/workspace/vessel/test/graypng'
ckpt_dir = '/daintlab/home/tmddnjs3467/workspace/checkpoint'
batch_size = 20

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

dataset_test = Dataset(os.path.join(test_dir), 
                        transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                          num_workers=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = UNet().to(device)
net = nn.DataParallel(net).cuda()
###부수적인 variable 생성
num_data_test = len(dataset_test)

#배치 몇개인지
num_batch_test = np.ceil(num_data_test / batch_size)

##네트워크 불러오는 함수
def load(ckpt_dir, net):

  dict_model = torch.load('%s/%s' % (ckpt_dir, 'model_epoch100(train20)(b4)(new2).pth'))

  net.load_state_dict(dict_model['net'])
  
  return net


fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_class = lambda x: 1.0 * (x > 0.5)

##test
import matplotlib.pyplot as plt
#이미지 저장할 폴더
result_dir = '/daintlab/home/tmddnjs3467/workspace/vessel/t20b4new2'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

st_epoch = 0
net = load(ckpt_dir=ckpt_dir, net=net)

id = 0

with torch.no_grad():
  net.eval()
  for batch, data in enumerate(loader_test, 1):
    input = data['input'].to(device)
   
    output = net(input)
    
    output = fn_tonumpy(fn_class(output))

    for j in range(output.shape[0]):
       
        ##png타입으로 저장하는 코드
        plt.imsave(os.path.join(result_dir, 'output_%02d.png' % id), output[j].squeeze(), cmap='gray')
        id += 1

print('output의 shape\n',output.shape)

##512512를 원본사이즈로 resize
for i in range(20):
  img_result = Image.open(os.path.join('%s' % result_dir, 'output_%02d.png' % i))
  
  img_result = img_result.resize((565, 584))

  result_ = np.asarray(img_result)

  plt.imsave(os.path.join(result_dir, 'output_%02d.png' % i), result_, cmap='gray')

