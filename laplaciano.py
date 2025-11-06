import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('CharizardPSA2.png', cv2.IMREAD_GRAYSCALE)
img_smooth = cv2.GaussianBlur(img, (3,3), 0)
img_hist = cv2.equalizeHist(img_smooth)

lap = cv2.Laplacian(img_hist, cv2.CV_64F)

img2 = cv2.imread('cha5.jpg', cv2.IMREAD_GRAYSCALE)
img_smooth2 = cv2.GaussianBlur(img2, (3,3), 0)
img_hist2 = cv2.equalizeHist(img_smooth2)

lap2 = cv2.Laplacian(img_hist2, cv2.CV_64F)

var_lap = lap.var()
print(f"Varianza del Laplaciano: {var_lap:.2f}")

var_lap2 = lap2.var()
print(f"Varianza del Laplaciano: {var_lap2:.2f}")

plt.figure(figsize=(10,4))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('Imagen original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(np.abs(lap), cmap='gray')
plt.title('Laplaciano (bordes)')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(img2, cmap='gray')
plt.title('Imagen original')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(np.abs(lap2), cmap='gray')
plt.title('Laplaciano (bordes)')
plt.axis('off')


plt.show()



