
####Final version for test#########(test1 is test dataset,test2 is train dataset)
#show 
#用plot显示
import h5py
import cv2
import numpy as np

#import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import datetime
import random


# input mudule file
import Py_autopilot2_module

from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import imageio
import scipy.io as sio 
import pandas as pd
from torchvision.io import read_image


from sklearn.utils import shuffle   # for loadFromPickle
from sklearn.model_selection import train_test_split    #for split
import math    #for pi

######### for CV && PIL##########
# 通过cv2加载的图片，默认色道格式为BGR，可通过cv2.cvtColor函数进行转换；
# 通过PIL加载的图片，默认色道格式为RGB，可通过图像的convert方法进行转换。
def load_img_by_cv(file):
    return cv2.imread(file)

def load_img_by_PIL(file):
    return Image.open(file)

def cv2PIL(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))

def PIL2cv(img_pil):
    return cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_RGB2BGR)
#################################

#打开交互模式(显示折线图)
plt.ion()


#超参设置
BATCH_SIZE = 128
EPOCH_NUM = 10

torch.manual_seed(0)  #固定每次的初始化值

#load data shuffle and spilit
features, labels = Py_autopilot2_module.loadFromPickle()   #shape:(45406, 100, 100) (45406,)
# features = features / 127.5 - 1.0       #-1 ~ 1     transforms.Tosensor已经0～255归一化为0～1

features = np.array(features).astype(np.uint8)   #把features的数据类型转化成unit8，作为tranform.ToTensor的输入

train_features, test_features, train_label, test_label = train_test_split(features, labels, test_size=0.3,random_state=0,shuffle=False)
train_features_for_test = train_features    # for test dataset_train 31784

# model = Py_autopilot2_module.AutoPilot2Net()


#load network
test_net = torch.load('models/model_5000.pkl')   #load network
test_net.eval() #指定test的网络


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
test_net = test_net.to(device)


# for test dataset_test=====================================================================



BATCH_SIZE_TEST =1
transform_test = transforms.Compose([
    transforms.ToTensor(),               ## shape:(31784, 100, 100, 3) -->(31784, 3,100, 100)
    transforms.Normalize((0.43567735,0.49298695,0.5192303), (0.22272868,0.24110594,0.29045662))     
])
dataset_test = Py_autopilot2_module.Custom_Dataset(
                                test_features,
                                test_label,
                                transform = transform_test
)
test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)




test_loss_rec = []
test_accr_rec=[]
net_out_rec=[]
steer_get_rec=[]

# for test picture
steer = cv2.imread('resources/steering_wheel_image.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0

for i, data in enumerate(test_dataloader):

    # 训练数据
    img_get, steer_get = data
    
    img_get, steer_get = img_get.to(device), steer_get.to(device)  #steer_get.shape = torch.Size([batchsize])

    net_out = test_net.forward(img_get).squeeze().float() 
    steer_get = steer_get.squeeze()

    # net_out = net_out.squeeze().float()

    #>>>>>>>>>>>>>>>>>>>>>>>>>显示3张图片，原始图，norm图，方向盘，第一排
    steer_pred_angle = net_out.detach().cpu().numpy() / math.pi * 180   # just for  steer angle picture 
    print("i=", i, "Pridect =", steer_pred_angle)

    #show transformed img
    img_transform = np.array(img_get.cpu().squeeze().numpy()).astype(np.uint8)  #3xWxH
    img_transform = np.swapaxes(img_transform,0,2)     #HxWx3
    img_transform = np.swapaxes(img_transform,0,1)     #WxHx3
    img_transform_rez = cv2.resize(img_transform, (cols, rows), interpolation=cv2.INTER_LINEAR)

    #show orginal img
    img_org = np.array(test_features[i]).astype(np.uint8)  #100x100x3
    img_org_rez = cv2.resize(img_org, (cols, rows), interpolation=cv2.INTER_LINEAR)
  

    #show steer angle
    #计算转角
    smoothed_angle += 0.2 * pow(abs((steer_pred_angle - smoothed_angle)), 2.0 / 3.0) * (
        steer_pred_angle - smoothed_angle) / abs(
        steer_pred_angle - smoothed_angle)
    #计算旋转矩阵
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -float(smoothed_angle), 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))

    #把单通道的图片变成3通道图片，每个通道一样即可
    dst_RGB = np.empty((cols, rows, 3), dtype = np.uint8) 
    dst_RGB[:,:,0]=dst
    dst_RGB[:,:,1]=dst
    dst_RGB[:,:,2]=dst

    img_all = np.hstack((img_org_rez, img_transform_rez, dst_RGB)) #水平堆叠图片

    ax1 = plt.subplot(3,1,1)
    ax1.set_title('INPUT & OUTPUT IMG')
    plt.imshow( cv2PIL(img_all))

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<显示3张图片，原始图，norm图，方向盘


    #为了画图
    net_out_rec.append(net_out.detach().cpu().numpy() / math.pi * 180 )
    steer_get_rec.append(steer_get.detach().cpu().numpy() / math.pi * 180)

    loss = test_net.steering_acc_loss(net_out.detach().cpu(),  steer_get.detach().cpu().float())  #double -> float , tensor

    loss_result = loss.detach().cpu().numpy() #???
    test_loss_rec.append(loss_result) # tensor强制转换成numpy

    #计算准确度accr值
    #精度=1 - |预测-真值|/pi   
    accr_temp= 1 - np.abs(steer_get.detach().cpu().numpy() - net_out.detach().cpu().numpy()) / math.pi
    
    test_accr_rec.append(accr_temp)

    #Display the Steer and Accuracy
    SUB_LEFT  = 0.125  # the left side of the subplots of the figure
    SUB_RIGHT = 0.9    # the right side of the subplots of the figure
    SUB_BOTTOM = 0.1   # the bottom of the subplots of the figure
    SUB_TOP = 0.9      # the top of the subplots of the figure
    SUB_WSPACE = 0.2   # the amount of width reserved for blank space between subplots,
                # expressed as a fraction of the average axis width
    SUB_HSPACE = 0.5   # the amount of height reserved for white space between subplots,
                # expressed as a fraction of the average axis height
    plt.subplots_adjust(hspace=SUB_HSPACE)
    ax2 = plt.subplot(3,1,2)
    ax2.set_title('VERIFY')
    if i < 100:
        x = np.array(range(0 , i+1))  # 0~i
        ax2.plot(x,net_out_rec[0:], color = 'red', label = 'Pred') #取i个
        ax2.plot(x,steer_get_rec[0:], color = 'blue', label = 'GD')
    else:
        x = np.array(range(i-99 , i + 1))  #i-99 ~ i 共100个数
        ax2.plot(x,net_out_rec[i-99 : i+1], color = 'red', label = 'Pred') #取100个   切片>= 开头 and < 结束
        ax2.plot(x,steer_get_rec[i-99 : i+1], color = 'blue', label = 'GD')

    plt.setp(ax2.get_xticklabels(), fontsize='small') # 6
    ax2.legend(loc='upper right',fontsize=6)

    #ACCR window
    ax3 = plt.subplot(3,1,3)
    test_accr_mean = np.mean(test_accr_rec)
    ax3.set_title('ACCURACY    accr_mean=:{}'.format(test_accr_mean))
    if i < 100:
        x = np.array(range(0 , i+1))  # 0~i
        ax3.plot(x,test_accr_rec[0:], color = 'red', label = 'accr') #取i个
    else:
        x = np.array(range(i-99 , i + 1))  #x00 ~ i
        ax3.plot(x,test_accr_rec[i-99 : i+1], color = 'red', label = 'accr') #取100个

    plt.setp(ax3.get_xticklabels(), fontsize='small') # 6
    ax3.legend(loc='upper right',fontsize=6)

    plt.pause(0.05)
    #清除当前画布
    plt.clf()

plt.ioff()              #给折线图

 

test_loss_mean = np.mean(test_loss_rec)
print("test loss mean =", test_loss_mean)
test_accr_mean = np.mean(test_accr_rec)
print("test accr mean =", test_accr_mean)


