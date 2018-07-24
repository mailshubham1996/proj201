
import os
import cv2
os.getcwd()
path = "C:/proj/images/train/"
for i, filename in enumerate(os.listdir(path)):
    image=cv2.imread(path+filename,0)
    #r = 100.0 / image.shape[1]
    #dim = (100, int(image.shape[0] * r))
    index=0
# perform the actual resizing of the image and show it
    s1=" "
    s1=filename
    index=s1.find(".")
    s1=s1[0:index]
   
    resized = cv2.resize(image, (255,255))
    cv2.imshow("resized", resized)
    s=" "
    s=path+filename
    #s=s+"r.png"
    cv2.imwrite(s,resized)
    cv2.waitKey(0)


    os.rename("C:/proj/images/train/" + filename, "C:/proj/images/train/" + s1 + ".png")
