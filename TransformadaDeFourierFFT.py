import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar la imagen en escala de grises
img = cv2.imread('CharizardPSA2.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("No se encontró la imagen 'pokemon_card.jpg'")

# 2. Calcular la Transformada de Fourier
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)  # Centrar las bajas frecuencias
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# 3. Mostrar imagen original y espectro
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

# 4. Filtro de frecuencias bajas (mantiene detalles grandes, elimina texturas)
rows, cols = img.shape
crow, ccol = rows//2 , cols//2
mask = np.zeros((rows, cols), np.uint8)
r = 30  # radio del círculo de bajas frecuencias
cv2.circle(mask, (ccol, crow), r, 1, -1)

# Aplicar la máscara
fshift_low = fshift * mask
img_low = np.fft.ifft2(np.fft.ifftshift(fshift_low))
img_low = np.abs(img_low)

# 5. Filtro de frecuencias altas (mantiene bordes y detalles finos)
mask_high = 1 - mask
fshift_high = fshift * mask_high
img_high = np.fft.ifft2(np.fft.ifftshift(fshift_high))
img_high = np.abs(img_high)

# 6. Mostrar resultados
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
