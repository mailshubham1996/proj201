from PIL import Image
import requests
from io import BytesIO
import cv2
url="http://172.27.234.118:8080/shot.jpg?rnd=574114"
path="C:/proj/images/test/"
counter=1
while True:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save("C:/proj/images/train/"+str(counter)+".jpg")
    counter+=1
    if ord('q')==cv2.waitKey(10):
        exit(0)
