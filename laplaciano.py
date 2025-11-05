import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 
script_dir = os.path.dirname(os.path.abspath(__file__))


img1_path = os.path.join(script_dir, 'CharizardPSA7.png')
img2_path = os.path.join(script_dir, 'CharizardPSA6.png')


img = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)


if img is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen. Revisa la ruta: {img1_path}")

img_blurred = cv2.GaussianBlur(img, (3, 3), 0)
img_eq = cv2.equalizeHist(img_blurred)

# 3. Calcular Laplaciano sobre la imagen pre-procesada
lap = cv2.Laplacian(img_eq, cv2.CV_64F)
var_lap = lap.var()
print(f"Varianza del Laplaciano (Img 1 - PSA2): {var_lap:.2f}")


img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Verificación para la imagen 2
if img2 is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen. Revisa la ruta: {img2_path}")

# 1. Reducir ruido
img2_blurred = cv2.GaussianBlur(img2, (3, 3), 0)

# 2. Normalizar la iluminación
img2_eq = cv2.equalizeHist(img2_blurred)

# 3. Calcular Laplaciano sobre la imagen pre-procesada
lap2 = cv2.Laplacian(img2_eq, cv2.CV_64F)
var_lap2 = lap2.var()
print(f"Varianza del Laplaciano (Img 2 - PSA8): {var_lap2:.2f}")

# --- Visualización (Ahora comparando las imágenes ecualizadas) ---
plt.figure(figsize=(10, 8))

plt.subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original (PSA2)')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title('Original (PSA8)')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(img_eq, cmap='gray')
plt.title('Ecualizada (PSA2)')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(img2_eq, cmap='gray')
plt.title('Ecualizada (PSA8)')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(np.abs(lap), cmap='gray')
plt.title('Laplaciano (PSA2)')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(np.abs(lap2), cmap='gray')
plt.title('Laplaciano (PSA8)')
plt.axis('off')

plt.tight_layout()
plt.show()