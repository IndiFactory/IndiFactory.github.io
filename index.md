---
title: Home
layout: home
---


import os
import glob
import time
import warnings
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib
from skimage.color import rgb2lab, lab2rgb
from fastai.data.external import untar_data, URLs

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from psutil import virtual_memory
import torch


gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('GPU 연결 실패!')
else:
  print(gpu_info)


ram_gb = virtual_memory().total / 1e9
print('{:.1f} gigabytes of available RAM\n'.format(ram_gb))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('학습을 진행하는 기기:',device)

coco_path =untar_data(URLs.COCO_SAMPLE) 

paths = glob.glob(str(coco_path)+ "/train_sample/*.jpg") 
print(coco_path)

class pix2pix_Generator(nn.Module): #역전파,순전파 사용가능
  def __init__(self):
    super().__init__()

    self.input_layer = nn.Sequential( #nn.ModuleList()
        nn.Conv2d(1, 64, kernel_size=4,stride=2,padding=1,bias=False)#3개의 채널 Lab 중에 L 하나만 넣기 때문에 1 임
    )

    self.encoder_1 = nn.Sequential(
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(64,128,kernel_size=4, stride=2,padding=1,bias=True),
        nn.BatchNorm2d(128)
    )
    self.encoder_2 =nn.Sequential(
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(128,256, Kenel_size=4, stride=2,padding=1,bias=True),
        nn.BatchNorm2d(256)
    )
    self.encoder_3 =nn.Sequential(
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(256,512, Kenel_size=4, stride=2,padding=1,bias=True),
        nn.BatchNorm2d(512)
    )
    self.encoder_4 =nn.Sequential(
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(512,512,Kenel_size=4, stride=2,padding=1,bias=True),
        nn.BatchNorm2d(512)
    )
    self.encoder_5 =nn.Sequential(
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(512,512,Kenel_size=4, stride=2,padding=1,bias=True),
        nn.BatchNorm2d(512)
    )
    self.encoder_6 =nn.Sequential(
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(512,512,Kenel_size=4, stride=2,padding=1,bias=True),
        nn.BatchNorm2d(512)
    )
    self.middle = nn.Sequential(
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(512,512,Kenel_size=4, stride=2,padding=1,bias=False),
        nn.ReLU(True),
        nn.ConvTranspose2d(512,512,kernel_size=4,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(512)
    )

    self.decoder_6 = nn.Sequential(
        nn.ReLU(True),
        nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride=2,padding=1, bias=False),#512가 두개니까
        nn.BatchNorm2d(512),
        nn.Dropout(0.5) #안써도 됨. 왜냐면 skip_connection으로 데이터가 통과되니까

    )

    self.decoder_5 = nn.Sequential(
        nn.ReLU(True),
        nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride=2,padding=1, bias=False),#직전레이어인decoder_6의 512와 encoder_5에서 skipconnection 된 512
        nn.BatchNorm2d(512),
        nn.Dropout(0.5)

    )
    self.decoder_4 = nn.Sequential(
        nn.ReLU(True),
        nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride=2,padding=1, bias=False),#직전레이어인decoder_5의 512와 encoder_4에서 skipconnection 된 512
        nn.BatchNorm2d(512)

    )
    self.decoder_3 = nn.Sequential(
        nn.ReLU(True),
        nn.ConvTranspose2d(1024, 256, kernel_size = 4, stride=2, padding=1, bias=False),#직전레이어인decoder_4의 512와 encoder_3에서 skipconnection 된 512
        nn.BatchNorm2d(256)

    )
    self.decoder_2 = nn.Sequential(
        nn.ReLU(True),
        nn.ConvTranspose2d(512, 128, kernel_size = 4, stride=2, padding=1, bias=False),#직전레이어인decoder_3의 input 256와 encoder_2에서 output 256
        nn.BatchNorm2d(128)

    )
    self.decoder_1 = nn.Sequential(
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 64, kernel_size = 4, stride=2, padding=1, bias=False),#직전레이어인decoder_3의 input 256와 encoder_2에서 output 256
        nn.BatchNorm2d(64)

    )
    self.output_layer=nn.Sequential(
        nn.ReLU(True),
        nn.ConvTranspose2d(64,2,kernel_size=4,stride=2,padding=1),
        nn.Tanh()
    )

  def forward(self,x):
    input_layer = self.input_layer(x)
    encoder_1=self.encoder_1(input_layer)
    encoder_2=self.encoder_2(encoder1)
    encoder_3=self.encoder_3(encoder2)
    encoder_4=self.encoder_4(encoder3)
    encoder_5=self.encoder_5(encoder4)
    encoder_6=self.encoder_6(encoder5)

    middle = self.middle(encoder_6)

    cat_6 = torch.cat((middle,encoder_6), dim=1)
    decoder_6 = self.decoder_6(cat_6)
    cat_5=torch.cat((decoder_6,encoder_5),dim=1)
    decoder_5 = self.decoder_5(cat_5)
    cat_4 = torch.cat((decoder_5,encoder_4),dim=1)
    decoder_4 = self.decoder_4(cat_4)
    cat_3 = torch.cat((decoder_4,encoder_3),dim=1)
    decoder_3 = self.decoder_3(cat_3)
    cat_2 = torch.cat((decoder_3,encoder_2),dim=1)
    decoder_2 = self.decoder_2(cat_2)
    cat_1 = torch.cat((decoder_2,encoder_1),dim=1)
    decoder_1 = self.decoder_1(cat_1)

    output_layer = self.output_layer(decoder_1)

    return output_layer
    
    
class pix2pix_Discriminator(nn.Module):
  def __init__(self): 
    super().__init__()

    self.model = nn.Sequential(
        nn.Conv2d(3,64, kernel_size = 4, stride = 2, padding=1, bias=False),
        nn.LeakyReLU(0.2,True),

        nn.Conv2d(64,128,kernel_size = 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2,True),

        nn.Conv2d(128,256, kernel_size = 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2,True),

        nn.Conv2d(256,512,kernel_size = 4, stride=2, padding=1, bias= False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2,True),

        nn.Conv2d(512,1,kernel_size = 4, stride = 2, padding=1, bias=False) 

    )
  def forward(self, x):
    return self.model(x)
    
    
    def init_weights(m):# m에 layer가 들어간다.
  if type(m) == nn.Conv2d:
    nn.init.normal_(m.weight.data, mean=0.0, std = 0.02)
    print("Convolution layer initialized!!!!")

  elif type(m) == nn.ConvTranspose2d:
    nn.init.normal_(m.weight.data, mean = 0.0,std=0.02)
    print("ConvTranspose initialized!!!!")
  elif type(m) == nn.BatchNorm2d:
    nn.init.normal_(m.wieght.data, mean=1., std = 0.02)
    nn.init.constant_(m.bias.data, 0.,)
    print("BatchNorm2d Initialized!!!!!")


def initialize_model(model):
  model.apply(init_weights)
  return model


def initialize_model(model):
  model.apply(init_weights)
  return model



class GANLoss(nn.Module):
  def __init__(self):
    super().__init()
    self.register_buffer('real_label',torch.tensor(1.0))
    self.register_buffer("fkake_label",torch.tensor(0.0))
    self.loss = nn.BCEWithLogistisLoss() # BCELoss+Sigmoid

  def get_lables(self, preds, target_is_real):
    if target_is_real:
      labels = self.real_label
    else:
      labels = self.fake_label
    return labels.expand_as(preds)

  def __call__(self, preds, target_is_real):
    labels = self.get_labels(preds, target_is_real)
    loss = self.loss(preds, labels)



def lab_to_rgb(L, ab):
  L = (L+1)*50
  ab = ab*110

  Lab = torch.cat([L,ab],dim=1)


  Lab = torch.cat([L,ab],dim=1).permute(0,2,3,1).cpu().numpy()


  rgb_imgs = []
  for img in Lab:
    rgb = lab2rgb(img)
    rgb_imgs.append(rgb)

  return np.stack(rgb_imgs, axis = 0)
  
  


model_generator = initialize_model(pix2pix_Generator())

#Discriminator 생성 후 초기화
model_discriminator = initialize_model(pix2pix_Discriminator())


model_generator.to(device)
model_discriminator.to(device)

#GANLoss 생성 후 GPU 로 넘기기
criterion = GANLoss().to(device)
#L1 loss 선언
L1 = nn.L1Loss()


optimizer_generator = optim.Adam(model_generator.parameters(), lr = 2e-4, betas=(0.5,0.999))

optimizer_discriminator = optim.Adam(model_discriminator.parameters(), lr = 2e-4, betas=(0.5,0.999))


epochs = 100  #1000번이 좋다.

for e  in range(epochs):
  for index, data in enumerate(tqdm(dataloader_train)):

    #L채널, ab채널 뽑아서 gpu로 넘기기
    L = data['L'].to(device)
    ab = data['ab'].to(device)

    #generator 에 L을 넣고, 가짜 색(fake color)를 얻어냅니다.
    fake_color = model_generator(L)
    #print(index)
