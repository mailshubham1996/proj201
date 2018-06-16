import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq,whiten
from sklearn.decomposition import PCA
os.getcwd()
path="C:/proj/images/train/"


#first make a dictionary
dictionarySize=20

BOW=cv2.BOWKMeansTrainer(dictionarySize)
sift=cv2.xfeatures2d.SIFT_create()
    
for i,filename in enumerate(os.listdir(path)):
    img1=cv2.imread(path+filename,0)
    kp,des=sift.detectAndCompute(img1,None)
    BOW.add(des)
FLANN_INDEX_KDTREE = 0
dictionary=BOW.cluster()

flann_params = dict(algorithm =FLANN_INDEX_KDTREE , trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {}) 
bow_extract = cv2.BOWImgDescriptorExtractor( sift , matcher )
bow_extract.setVocabulary(dictionary) # the 64x20 dictionary, you made before

traindata=[]
trainlabels=[]
s=" "
index=0
for i ,filename in enumerate(os.listdir(path)):
    img1=cv2.imread(path+filename,0)
    siftkp=sift.detect(img1)
    bowsig=bow_extract.compute(img1,siftkp)
    traindata.extend(bowsig)
    s=" "
    s=filename
    index=s.find(".")
    s=s[0:index]
    s=int(float(s))
    if s<=1000:# means label of diseased leaf
        trainlabels.append(0)
    elif s>1000:
        trainlabels.append(1)

svm=cv2.ml.SVM_create()
svm.train(np.array(traindata),cv2.ml.ROW_SAMPLE,np.array(trainlabels))


path2="C:/proj/images/test/test3.jpg"

imgtest=cv2.imread(path2,0)
siftkp2=sift.detect(imgtest,None)
bowsig=bow_extract.compute(imgtest,siftkp2)

#print(traindata)
p=svm.predict(bowsig)
print(p)























#print('*'*40)
#3print(cen)    
