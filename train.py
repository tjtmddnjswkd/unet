from unet import UNet
from dataset2 import Dataset
from transform import Normalization, RandomFlip, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import numpy as np
# from tensorflow.keras.callbacks import TensorBoard
# from torch.utils.tensorboard import SummaryWriter

lr = 0.001
batch_size = 4
num_epoch = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '/daintlab/home/tmddnjs3467/workspace/vessel'
ckpt_dir = '/daintlab/home/tmddnjs3467/workspace/checkpoint'
log_dir = '/daintlab/home/tmddnjs3467/workspace/log'

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(),
                                ToTensor()])
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train(png20)'), 
                        transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                          num_workers=8)

# dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val(png)'), 
#                       transform=transform)
# loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, 
#                         num_workers=8)
print('input의 shape는\n', dataset_train[0]['input'].shape)

print('label의 shape는\n', dataset_train[0]['label'].shape)

##네트워크 생성하기
##학습이 되는 도메인이 CPU인지 GPU인지 확인하기위해 to어쩌구씀
net = UNet().to(device)
net = nn.DataParallel(net).cuda()

##loss function define
fn_loss = nn.BCEWithLogitsLoss().to(device)

##optimizer define
optim = torch.optim.Adam(net.parameters(), lr=lr)

###부수적인 variable 생성
num_data_train = len(dataset_train)
# num_data_val = len(dataset_val)

#배치 몇개인지
num_batch_train = np.ceil(num_data_train / batch_size)
# num_batch_val = np.ceil(num_data_val / batch_size)

##부수적인 function 생성
#텐서를 넘파이로
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
#디노말라이즈
fn_denorm = lambda x, mean, std: (x * std) + mean
#클래스분류
fn_class = lambda x: 1.0 * (x > 0.5)

##Tensorboard를 사용하기 위한 SummaryWriter 설정

# writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
# writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

##네트워크 저장하는 함수
def save(ckpt_dir, net, optim, epoch):
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
              '%s/model_epoch%d(train20)(b4)(new2).pth' % (ckpt_dir, epoch))

##네트워크 불러오는 함수
def load(ckpt_dir, net, optim):
  
  dict_model = torch.load('%s/%s' % (ckpt_dir, 'model_epoch100(train20)(b4)(new).pth'))

  net.load_state_dict(dict_model['net'])
  optim.load_state_dict(dict_model['optim'])
  ##미세조정해서 쓰기위해 에폭 조절.
  epoch = 0
  return net, optim, epoch

def iouscore(output, label):
  u_arr = np.zeros((2,512,512,1))
  print(u_arr.shape)
  for k in range(2):
    for i in range(512):
      for j in range(512):
        if(label[k][i][j][0]==1 or output[k][i][j][0]==1):
          u_arr[k][i][j][0]=1
  i_arr = output * label

  iou_arr = []
  for i in range(2):
    iou_arr.append(i_arr[i].sum()/u_arr[i].sum())
  iou_arr = np.array(iou_arr)
  return iou_arr.mean()


# net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

st_epoch = 0
for epoch in range(st_epoch + 1, num_epoch + 1):
  ##네트워크에게 트레인이라는 것을 알려줌
  net.train()
  loss_arr = []
  for batch, data in enumerate(loader_train, 1):
    # forward pass
    label = data['label'].to(device)
    input = data['input'].to(device)

    output = net(input)

    # backward pass
    optim.zero_grad()

    loss = fn_loss(output, label)
    loss.backward()
  
    optim.step()

    # 손실함수 계산
    loss_arr += [loss.item()]
    print('TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f ' %
          (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
  #   # Tensorboard 저장하기
  #   label = fn_tonumpy(label)
  #   input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
  #   output = fn_tonumpy(fn_class(output))
  #   #라벨 이미지 아웃풋 영상을 텐서보드에 작성하는 부분
  #   writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
  #   writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
  #   writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
  # #로스를 작성
  # writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

  # 백프로파게이션 validation과정에선 필요없음
  # with torch.no_grad():
  #   net.eval()
  #   loss_arr = []
  #   dice_arr = []
  #   for batch, data in enumerate(loader_val, 1):
  #     # forward pass
  #     label = data['label'].to(device)
  #     input = data['input'].to(device)

  #     output = net(input)

  #     #손실함수
  #     loss = fn_loss(output, label)

  #     loss_arr += [loss.item()]
  #     print('VAL: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f ' %
  #         (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))
      # # Tensorboard 저장하기
      # label = fn_tonumpy(label)
      # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
      # output = fn_tonumpy(fn_class(output))
      # #라벨 이미지 아웃풋 영상을 텐서보드에 작성하는 부분
      # writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
      # writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
      # writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
    # # 로스를 작성
    # writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
  if epoch % 100 == 0:
    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)



