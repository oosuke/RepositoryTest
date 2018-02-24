# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from deep_convnet import DeepConvNet
from PIL import Image
from common.functions import softmax

size = 28

test = np.zeros((1,1,size,size))
img = Image.open("../dataset/test.png")
img = img.resize((size,size))
img_rgb = img.convert('RGB')

for x in range(size):
    for y in range(size):
        r,g,b = img_rgb.getpixel((x,y))
        test[0,0,y,x] = g

network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")

print(np.argmax(softmax(network.predict(test))))