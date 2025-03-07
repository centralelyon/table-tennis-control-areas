import cv2
import numpy as np
from PIL import Image, ImageDraw

def detect_and_draw_contours(image_path, output_path):
    # Ouvrir l'image avec Pillow
    image = Image.open(image_path).convert("RGB")

    # Convertir l'image Pillow en tableau NumPy
    image_np = np.array(image)

    # Convertir en image en niveaux de gris
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Appliquer un flou gaussien pour réduire le bruit
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Détecter les contours avec Canny
    edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)

    # Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convertir l'image originale en une version modifiable pour dessiner
    image_draw = ImageDraw.Draw(image)

    # Dessiner les contours sur l'image
    for contour in contours:
        print(contour)
        # Convertir les contours en une liste de tuples
        contour_points = [(point[0][0], point[0][1]) for point in contour]
        image_draw.line(contour_points, fill=(0, 255, 0), width=2)

    # Enregistrer l'image avec les contours dessinés
    image.save(output_path)
    print(f"Image saved at {output_path}")

# Exemple d'utilisation
detect_and_draw_contours('C:/Users/ReViVD/Documents/GitHub/tt-espace/output/match_ALEXIS-LEBRUN_vs_MA-LONG/heatmap/binaire_heatmap_frame_0.png', 'output_image_with_contours.png')
