import cv2
import numpy as np

# Création d'une image en uint16 pour éviter la saturation immédiate
image = np.zeros((100, 100, 3), dtype=np.uint16)

# Définition d'un contour carré (5x5 pixels) centré en (50,50)
contour = np.array([[[45, 45]], [[55, 45]], [[55, 55]], [[45, 55]]], dtype=np.int32)

# Dessiner directement sur l'image 5 fois
for _ in range(5):
    cv2.drawContours(image, [contour], -1, (0, 10, 0), thickness=-1)  

# Convertir l'image en uint8 (normalisation si besoin)
image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

# Affichage de la valeur finale du pixel (50,50)
print("Valeur finale du pixel (50,50) :", image[50, 50])
