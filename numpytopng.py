import numpy as np
from PIL import Image
import os
import cv2

##데이터셋 png파일로 다 저장하기
def numpy_to_png(source, target, num):
    for i in range(num):
        img_input = np.load(os.path.join(source, 'input_%02d.npy' % i))
        img_label = np.load(os.path.join(source, 'label_%02d.npy' % i))

        # img_input = img_input.resize((512, 512))
        # img_label = img_label.resize((512, 512))

        # input_ = np.asarray(img_input)
        # label_ = np.asarray(img_label)

        image = Image.fromarray(img_input)
        label = Image.fromarray(img_label)

        image.save(os.path.join(target, 'input_%02d.png' % i))
        image.save(os.path.join(target, 'label_%02d.png' % i))

def rgb_to_gray(source, target):
    imagePaths = [os.path.join(source, file_name) for file_name in os.listdir(source)]
    for imagePath in imagePaths:
        img = Image.open(imagePath).convert('L')
        img_numpy = np.array(img, 'uint8')
        cv2.imwrite("%s/" % target + imagePath.split("/")[-1], img_numpy)
    print("All Done")



def main():
    source = '/daintlab/home/tmddnjs3467/workspace/vessel/train(png80)'
    target = '/daintlab/home/tmddnjs3467/workspace/vessel/train(graypng80)'
    num = 80
    # numpy_to_png(source, target, num)
    rgb_to_gray(source, target)
    
    
if __name__ == '__main__':
    main()