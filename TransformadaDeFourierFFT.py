import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- INICIO DE LA CORRECCIÓN DE RUTA ---
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, 'CharizardPSA2.png')
# --- FIN DE LA CORRECCIÓN DE RUTA ---

# 4. Cargar la imagen en escala de grises
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 5. Verificación
if img is None:
    raise FileNotFoundError(f"No se encontró la imagen. Revisa la ruta: {img_path}")

# === INICIO DE LA CORRECCIÓN DE PROCESAMIENTO ===
# 6. Pre-procesamiento para normalizar brillo y reducir ruido
img_blurred = cv2.GaussianBlur(img, (3, 3), 0)
img_eq = cv2.equalizeHist(img_blurred)
# === FIN DE LA CORRECCIÓN DE PROCESAMIENTO ===

# 7. Calcular la Transformada de Fourier
f = np.fft.fft2(img_eq)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# 8. Mostrar imagen original, ecualizada y espectro
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen original (sin procesar)')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(img_eq, cmap='gray')
plt.title('Imagen Normalizada (Ecualizada)')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Espectro de magnitud (Fourier)')
plt.axis('off')
plt.tight_layout()
plt.show()

# 9. Filtro de frecuencias bajas
rows, cols = img_eq.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
r = 30
cv2.circle(mask, (ccol, crow), r, 1, -1)
fshift_low = fshift * mask
img_low = np.fft.ifft2(np.fft.ifftshift(fshift_low))
img_low = np.abs(img_low)

# 10. Filtro de frecuencias altas
mask_high = 1 - mask
fshift_high = fshift * mask_high
img_high = np.fft.ifft2(np.fft.ifftshift(fshift_high))
img_high = np.abs(img_high)

# === INICIO DE LA MODIFICACIÓN SOLICITADA ===
#
# 11. Calcular y mostrar las varianzas de las imágenes filtradas
#
var_low_freq = img_low.var()
var_high_freq = img_high.var()

print(f"Varianza de Bajas Frecuencias (Fourier): {var_low_freq:.2f}")
print(f"Varianza de Altas Frecuencias (Fourier): {var_high_freq:.2f}")
# === FIN DE LA MODIFICACIÓN SOLICITADA ===


# 12. Mostrar resultados de los filtros
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(img_eq, cmap='gray')
plt.title('Normalizada (Referencia)')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(img_low, cmap='gray')
plt.title('Bajas frecuencias (suave)')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(img_high, cmap='gray')
plt.title('Altas frecuencias (bordes)')
plt.axis('off')
plt.tight_layout()
plt.show()