from unet import UNet
from dataset2 import Dataset
from dataset import Dataset_train
from transform import Normalization, RandomFlip, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import numpy as np
import wandb
import argparse
from measurement import iouscore, accuracy

# import imgcat
# 1. Start a W&B run
wandb.init(project="Retina vessel data segmentation")
parser = argparse.ArgumentParser()
parser.add_argument('--lr',
                type=float,
                default=0.001,
                help='learning rate')   
parser.add_argument('--epochs',
                type=int,
                default=100,
                help='num epochs')
parser.add_argument('--train_batch',
                type=int,
                default=8,
                help='train batch size')
parser.add_argument('--val_batch',
                type=int,
                default=4,
                help='validation batch size')
parser.add_argument('--model_name',
                type=str,
                default='SAaug',
                help='model name')
parser.add_argument('--train_data_set',
                type=str,
                default='train(592png16)',
                help='input data path')
parser.add_argument('--test_data_set',
                type=str,
                default='val(592png4)',
                help='test data path')
parser.add_argument('--gpu',
                type=str,
                default='0',
                help='gpu number')

args = parser.parse_args()
wandb.config.update(args)

# lr = 0.001
# batch_size = 4
# num_epoch = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '/daintlab/home/tmddnjs3467/workspace/vessel'
ckpt_dir = '/daintlab/home/tmddnjs3467/workspace/checkpoint'
log_dir = '/daintlab/home/tmddnjs3467/unet'

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(),
                                ToTensor()])
dataset_train = Dataset(data_dir=os.path.join(data_dir, args.train_data_set), 
                        transform=transform)
loader_train = DataLoader(dataset_train, batch_size=args.train_batch, shuffle=True,
                          num_workers=8)

dataset_val = Dataset(data_dir=os.path.join(data_dir, args.test_data_set), 
                      transform=transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()]))
loader_val = DataLoader(dataset_val, batch_size=args.val_batch, shuffle=False, 
                        num_workers=8)
print('input의 shape는\n', dataset_train[0]['input'].shape)

print('label의 shape는\n', dataset_train[0]['label'].shape)

##네트워크 생성하기
##학습이 되는 도메인이 CPU인지 GPU인지 확인하기위해 to어쩌구씀
net = UNet().to(device).cuda()
net = nn.DataParallel(net)

##loss function define
fn_loss = nn.BCEWithLogitsLoss().to(device)
# iou_loss = mIoULoss().to(device)
##optimizer define
optim = torch.optim.Adam(net.parameters(), lr=args.lr)

###부수적인 variable 생성
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

#배치 몇개인지
num_batch_train = np.ceil(num_data_train / args.train_batch)
num_batch_val = np.ceil(num_data_val / args.val_batch)

##부수적인 function 생성
#텐서를 넘파이로
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
#디노말라이즈
fn_denorm = lambda x, mean, std: (x * std) + mean
#클래스분류
fn_class = lambda x: 1.0 * (x > 0.5)


##네트워크 저장하는 함수
def save(ckpt_dir, net, optim, epoch, loss):
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
              '%s/model_epoch%d_loss%.3f(592)%s.pth' % (ckpt_dir, epoch, loss, args.model_name))

# ##네트워크 불러오는 함수
# def load(ckpt_dir, net, optim):
  
#   dict_model = torch.load('%s/%s' % (ckpt_dir, 'model_epoch100(512)(filp+gray).pth'))
#   net.load_state_dict(dict_model['net'])
#   optim.load_state_dict(dict_model['optim'])
#   ##미세조정해서 쓰기위해 에폭 조절.
#   epoch = 0
#   return net, optim, epoch

# net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

st_epoch = 0
for epoch in range(st_epoch + 1, args.epochs + 1):
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
    # loss = iou_loss(output, label)
    loss.backward()
  
    optim.step()
    label = fn_tonumpy(label)
    output = fn_tonumpy(fn_class(output))
    # 손실함수 계산
    loss_arr += [loss.item()]
    print('TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f | IOU %.4f | ACC %.4f' %
          (epoch, args.epochs, batch, num_batch_train, np.mean(loss_arr), iouscore(output, label), accuracy(output, label)))
  dataset_train = Dataset(data_dir=os.path.join(data_dir, args.train_data_set), 
                      transform=transform)
  loader_train = DataLoader(dataset_train, batch_size=args.train_batch, shuffle=True,
                        num_workers=8)

  #wandb 저장
  wandb.log({"train loss": np.mean(loss_arr)})
  # 백프로파게이션 validation과정에선 필요없음
  with torch.no_grad():
    net.eval()
    loss_arr = []
    iou_arr = []
    for batch, data in enumerate(loader_val, 1):
      # forward pass
      label = data['label'].to(device)
      input = data['input'].to(device)

      output = net(input)

      #손실함수
      loss = fn_loss(output, label)
      # loss = iou_loss(output, label)
      loss_arr += [loss.item()]
      #원래 이미지들
      label = fn_tonumpy(label)
      input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
      output = fn_tonumpy(fn_class(output))
      print('VAL: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f | IOU %.4f | ACC %.4f' %
          (epoch, args.epochs, batch, num_batch_val, np.mean(loss_arr), iouscore(output, label), accuracy(output, label)))
      iou_arr += [iouscore(output, label)]
    # 로그 작성
    wandb.log({"input": [wandb.Image(input[3], caption="input")], "label": [wandb.Image(label[3], caption="label")],
                "output": [wandb.Image(output[3], caption="output")]})
    wandb.log({"val loss": np.mean(loss_arr)})
    wandb.log({"val iou": np.mean(iou_arr)})
  if epoch % args.epochs == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch, loss=np.mean(loss_arr))



