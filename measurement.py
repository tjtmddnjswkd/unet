import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

def iouscore(output, label):
    u_arr = np.logical_or(output, label)
    i_arr = output * label
    iou_arr = []
    for i in range(4):
        iou_arr.append(i_arr[i].sum()/u_arr[i].sum())
    iou_arr = np.array(iou_arr)
    return iou_arr.mean()

def accuracy(output, label):
    acc = (output == label)
    return acc.mean()

def main():
    print(123)

if __name__ == '__main__':
    main()