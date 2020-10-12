import os
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 

data_dir = '/daintlab/data/seungwon/vessel'

def change_filename(type, address):
    #원래 training images 파일 이름들 확인
    lst_name = os.listdir(address)
    lst_name.sort()
    print('파일 이름들 : \n', lst_name)

    #training이미지들 이름 바꿔주기.
    i = 0
    for name in lst_name:
        src = os.path.join(address, name)
        dst = '%02d.%s' % (i, type)
        dst = os.path.join(address, dst)
        os.rename(src, dst)
        i += 1
    
    lst_name = os.listdir(address)
    lst_name.sort()
    print('바뀐 파일 이름들 : \n', lst_name)

#파일이름 변경완료 01.tif ~ 19.tif, 01.gif ~ 19.gif 
###이거는 맨처음 한번만 돌리면 됨
# change_filename('tif', '%s/training/images' % data_dir)
# change_filename('gif', '%s/training/1st_manual' % data_dir)

#데이터 size확인
img_label = Image.open(os.path.join('%s/training/1st_manual' % data_dir, '01.gif'))
img_input = Image.open(os.path.join('%s/training/images' % data_dir, '01.tif'))

ny, nx = img_label.size 

ny2, nx2 = img_input.size
 
print('label의 사이즈 (y, x)순\n', ny, nx)
print('input의 사이즈 (y, x)순\n', ny2, nx2)

#training, validation 을 random하게 나누기 위한 id
id_num = np.arange(20)
np.random.shuffle(id_num)
id_num

#나눈 데이터셋에 대한 폴더 생성
dir_save_train = os.path.join(data_dir, 'train(16)')
dir_save_val = os.path.join(data_dir, 'val(4)')

if not os.path.exists(dir_save_train):
  os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
  os.makedirs(dir_save_val)

def convert_image_to_numpy(start, end, size, type):
    num = start
    for i in range(start,end):
        img_label = Image.open(os.path.join('%s/training/1st_manual' % data_dir, '%02d.gif' % id_num[i]))
        img_input = Image.open(os.path.join('%s/training/images' % data_dir, '%02d.tif' % id_num[i]))
        
        img_label = img_label.resize((size, size))
        img_input = img_input.resize((size, size))

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        ##이건 augmentation
        # label_ = np.flipud(label_)
        # input_ = np.flipud(input_)
        # label_ = np.fliplr(label_)
        # input_ = np.fliplr(input_)
        if type == 'train':
            np.save(os.path.join(dir_save_train, 'label_%02d.npy' % (num) ), label_)
            np.save(os.path.join(dir_save_train, 'input_%02d.npy' % (num) ), input_)
        elif type == 'val':
            np.save(os.path.join(dir_save_val, 'label_%02d.npy' % (num) ), label_)
            np.save(os.path.join(dir_save_val, 'input_%02d.npy' % (num) ), input_)
        else:
            print("좋지 않은 타입입니다.")        
        
        num += 1
    
    return label_, input_
# #얘네는 training val 바꾸고 싶을 때 실행시키면 됨.
# convert_image_to_numpy(0, 16, 512, 'train')
# label_, input_ = convert_image_to_numpy(16, 20, 512, 'val')

#test 데이터들 numpy로 변환
if not os.path.exists('%s/test/numpy' % data_dir):
    os.makedirs('%s/test/numpy' % data_dir)

for i in range(1,21):
        img_input = Image.open(os.path.join('%s/test/images' % data_dir, '%02d_test.tif' % i))
        
        img_input = img_input.resize((512, 512))

        input_ = np.asarray(img_input)

        np.save(os.path.join('%s/test/numpy' % data_dir, 'input_%02d.npy' % (i) ), input_)

# ##실제 input label 확인
# plt.subplot(121)
# plt.imshow(label_, cmap='gray')
# plt.title('label')

# plt.subplot(122)
# plt.imshow(input_, cmap='gray')
# plt.title('input')
# plt.show()
