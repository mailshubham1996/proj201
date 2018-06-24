#PCA ASSESSMENT
'''import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq,whiten
from sklearn.decomposition import PCA
os.getcwd()
path="C:/proj/images/train/"
desc=[]

for i,filename in enumerate(os.listdir(path)):

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
    pca=PCA(n_components=128)
    pca.fit(x)
    print(pca.singular_values_)
    #print(np.shape(des1))
    #centroids=centroids.transpose()
        
        #np.concatenate((cen,centroids),axis=0)


#WATERSHED ALGORITHM
import numpy as np
import cv2
from matplotlib import pyplot as plt
path="C:/proj/images/test/64.png"
img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#oise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3) 
# Finding sure foreground area
dist_transform = opening
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0) 
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)



ret, markers = cv2.connectedComponents(sure_fg) 
# Add one to all labels so that sure background is not 0, but 1
markers = markers
# Now, mark the region of unknown with zero
markers[unknown==255] = 0




markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]


cv2.imshow(' ',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''


#grab cut

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

path="C:/proj/images/test/65.png"
img = cv.imread(path)
kret=img


mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)


rect = (10,10,600,400)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]


plt.imshow(img)

plt.colorbar()
plt.show()




