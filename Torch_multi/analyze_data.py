#coding=utf8
import numpy as np
import torch
from sklearn.decomposition.pca import PCA
import matplotlib
import matplotlib.pyplot as plt

cc=torch.load('params/param_mix101_WSJ0_emblayer_180')
cc=cc['layer.weight'].cpu().numpy()
pca=PCA(2)
newData=pca.fit_transform(cc)
print newData.shape
f1=plt.figure(1)
for data in newData:
    plt.scatter(newData[:,0],newData[:,1])
plt.show()

