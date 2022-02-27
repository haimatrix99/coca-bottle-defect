import cv2
import numpy as np

def img_to_np(img):  
    img_array = []
    img = cv2.resize(img, (64,64))
    img_array.append(np.asarray(img))
    images = np.array(img_array)
    return images
