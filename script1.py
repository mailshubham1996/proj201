import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

os.getcwd()
path="C:/proj/images/train/"
desc=[]
for i,filename in enumerate(os.listdir(path)):
# Initiate SIFT detector for each image
    sift = cv2.xfeatures2d.SIFT_create()
    img1=cv2.imread(path+filename,0)
  #  img1=(img1).astype(np.uint8)
# find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    #desc.append(des1)
    print(des1,filename)
    
#now cluseter the set of descriptors using k means
#print(desc)
desc=np.array(desc)

print(desc)

##write desc to txt file and add label to each of the desc rows 0 for diseased and 1 fornot
