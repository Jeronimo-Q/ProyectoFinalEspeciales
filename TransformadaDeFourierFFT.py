import cv2
import numpy as np
import matplotlib.pyplot as plt
#CharizardPSA2.png
img = cv2.imread('CharizardPSA2.png', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

img_smooth = cv2.GaussianBlur(img, (3,3), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_hit = clahe.apply(img_smooth)
img_norm = cv2.normalize(img_hit, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

f = np.fft.fft2(img_norm)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Imagen original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Espectro de magnitud (Fourier)')
plt.axis('off')

plt.tight_layout()
plt.show()


crow, ccol = rows//2 , cols//2
mask = np.zeros((rows, cols), np.uint8)
r = int(0.05 * min(rows, cols)) 
cv2.circle(mask, (ccol, crow), r, 1, -1)

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Imagen original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title('mascara de paso bajo')
plt.axis('off')

plt.tight_layout()
plt.show()

fshift_low = fshift * mask
img_low = np.fft.ifft2(np.fft.ifftshift(fshift_low))
img_low = np.abs(img_low)

r = int(0.10 * min(rows, cols)) 
cv2.circle(mask, (ccol, crow), r, 1, -1)
feshift_mid = fshift * mask
img_mid = np.fft.ifft2(np.fft.ifftshift(feshift_mid))
img_mid = np.abs(img_mid)

mask_high = 1 - mask
fshift_high = fshift * mask_high
img_high = np.fft.ifft2(np.fft.ifftshift(fshift_high))
img_high = np.abs(img_high)

plt.subplot(1,2,1)
plt.imshow(img_mid, cmap='gray')
plt.title('Imagen original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title('mascara de paso bajo')
plt.axis('off')

plt.tight_layout()
plt.show()


energy_high = np.sum(np.abs(fshift_high)) / (rows * cols)
energy_low = np.sum(np.abs(fshift_low)) / (rows * cols)
energy_mid = np.sum(np.abs(feshift_mid)) / (rows * cols)
print("Frecuencias altas: ",energy_high, "Frecuencias bajas: ",energy_low)
print("Calidad visual frecuencias altas vs bajas: ",round( (energy_high*0.6 + energy_mid*0.4)/(energy_low+energy_mid+energy_high),4))

plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(img_low, cmap='gray')
plt.title('Bajas frecuencias (suave)')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(img_high, cmap='gray')
plt.title('Altas frecuencias (bordes)')
plt.axis('off')

plt.tight_layout()
plt.show()