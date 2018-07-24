
# coding: utf-8

# In[6]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq,whiten
from sklearn.decomposition import PCA
from sklearn.svm import SVC
os.getcwd()


# In[27]:


path="C:/proj/images/train/"

dictionarySize=300


# In[28]:


BOW=cv2.BOWKMeansTrainer(dictionarySize)#initailzes bag of words with k means algorithm as model
sift=cv2.xfeatures2d.SIFT_create()


# In[29]:


for i,filename in enumerate(os.listdir(path)):
    img1=cv2.imread(path+filename,0)
    kp,des=sift.detectAndCompute(img1,None)
    BOW.add(des)##creating bag of words and adding the sift features
    


# In[30]:


FLANN_INDEX_KDTREE = 0
dictionary=BOW.cluster()#for every image feature we construct the image descriptor



# In[31]:


flann_params = dict(algorithm =FLANN_INDEX_KDTREE , trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {}) 


# In[32]:


bow_extract = cv2.BOWImgDescriptorExtractor( sift , matcher )
bow_extract.setVocabulary(dictionary) 


# In[33]:


traindata=[]
trainlabels=[]
s=" "
index=0


# In[34]:


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
    if s<1000:# means label of diseased leaf
        trainlabels.append(0)
    elif s>=1000:
        trainlabels.append(1)


# In[70]:


svm=cv2.ml.SVM_create()
svm.train(np.array(traindata),cv2.ml.ROW_SAMPLE,np.array(trainlabels))#x_train y_train


path2="C:/proj/images/test/55.png"


# In[71]:


imgtest=cv2.imread(path2,0)
imgtest = cv2.resize(imgtest, (255,255))
siftkp2=sift.detect(imgtest,None)
bowsig=bow_extract.compute(imgtest,siftkp2)


# In[72]:


p=svm.predict(bowsig)
svm2=SVC(probability=True)
svm2.fit(np.array(traindata),np.array(trainlabels))


# In[73]:


q=svm2.predict_proba(bowsig)

print(q)


#aim is to make least false negative predictions from svm

