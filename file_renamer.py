import os
os.getcwd()
collection = "C:/proj/images/train"
for i, filename in enumerate(os.listdir(collection)):
    os.rename("C:/proj/images/train/" + filename, "C:/proj/images/train/" + str(i) + ".png")
    image=cv2.imread(path+filename,0)
    #r = 100.0 / image.shape[1]
    #dim = (100, int(image.shape[0] * r))
 
    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, (255,255))
   #added the resized part.
    cv2.waitKey(0)

              
              
              
              
              
              
              
