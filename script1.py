import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq,whiten
from sklearn.decomposition import PCA
os.getcwd()
path="C:/proj/images/train/"
desc=[]
f = open('C:/proj/dataset.txt', 'wb')
cen=np.zeros([1,3],dtype=int)
for i,filename in enumerate(os.listdir(path)):
e
    sift = cv2.xfeatures2d.SIFT_create()
    img1=cv2.imread(path+filename,0)
  #  img1=(img1).astype(np.uint8)
# find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    #desc.append(des1)
    #print(des1,filename)    
    #now cluseter the set of descriptors using k means
    #to have a constant number of vectors
    x=des1
    pca=PCA(n_components=4)
    pca.fit(x)
    print(pca.singular_values_)
     
    #centroids=centroids.transpose()
        
        #np.concatenate((cen,centroids),axis=0)
    print(centroids)
print(centroids.shape)#it should be (1,128)
print('*'*40)
print(cen)    

    


##write desc to txt file and add label to each of the desc rows 0 for diseased and 1 fornot


