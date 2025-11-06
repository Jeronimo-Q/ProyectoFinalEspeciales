import numpy as np
import matplotlib.pyplot as plt 
import cv2 


img = cv2.imread('cha5.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

img_smooth = cv2.GaussianBlur(img, (3,3), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_hit = clahe.apply(img_smooth)
img_norm = cv2.normalize(img_hit, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


